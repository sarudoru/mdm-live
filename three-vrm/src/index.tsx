import React, { Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import './styles.css'
import { GLTF, GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import { VRM, VRMLoaderPlugin, VRMHumanBoneName } from '@pixiv/three-vrm'
import { Object3D, Quaternion, Vector3 } from 'three'
import { useControls } from 'leva'

// Lightweight redirect for dev server: if user opens /motion-test on :3000,
// send them to the static test page that lives in /public.
if (typeof window !== 'undefined') {
  const p = window.location.pathname.replace(/\/$/, '')
  if (p === '/motion-test') {
    const { origin } = window.location
    // If we're on the React dev server (likely :3000), redirect to the FastAPI backend (:8000)
    if (/:(3000)$/.test(origin)) {
      const target = origin.replace(/:3000$/, ':8000') + '/motion-test'
      window.location.replace(target)
    } else {
      // Otherwise, try local static fallback
      window.location.replace('/motion-test.html')
    }
  }
}

type MotionFrame = {
  root_position: number[]
  rotations: number[][]
  positions?: number[][]
}

type MotionData = {
  version: number
  fps: number
  joints: string[]
  frames: MotionFrame[]
}

type SegmentState = {
  description: string
  duration_seconds: number
  motionUrl: string
  motionData?: MotionData
}

const API_BASE = process.env.REACT_APP_API_BASE ?? ''

const HML_TO_VRM: Record<string, VRMHumanBoneName | null> = {
  pelvis: VRMHumanBoneName.Hips,
  left_hip: VRMHumanBoneName.LeftUpperLeg,
  right_hip: VRMHumanBoneName.RightUpperLeg,
  spine1: VRMHumanBoneName.Spine,
  left_knee: VRMHumanBoneName.LeftLowerLeg,
  right_knee: VRMHumanBoneName.RightLowerLeg,
  spine2: VRMHumanBoneName.Chest,
  left_ankle: VRMHumanBoneName.LeftFoot,
  right_ankle: VRMHumanBoneName.RightFoot,
  spine3: VRMHumanBoneName.UpperChest,
  left_foot: VRMHumanBoneName.LeftToes,
  right_foot: VRMHumanBoneName.RightToes,
  neck: VRMHumanBoneName.Neck,
  left_collar: VRMHumanBoneName.LeftShoulder,
  right_collar: VRMHumanBoneName.RightShoulder,
  head: VRMHumanBoneName.Head,
  left_shoulder: VRMHumanBoneName.LeftUpperArm,
  right_shoulder: VRMHumanBoneName.RightUpperArm,
  left_elbow: VRMHumanBoneName.LeftLowerArm,
  right_elbow: VRMHumanBoneName.RightLowerArm,
  left_wrist: VRMHumanBoneName.LeftHand,
  right_wrist: VRMHumanBoneName.RightHand,
}

const TRACKED_BONES: VRMHumanBoneName[] = [
  VRMHumanBoneName.Hips,
  VRMHumanBoneName.Spine,
  VRMHumanBoneName.Chest,
  VRMHumanBoneName.UpperChest,
  VRMHumanBoneName.Neck,
  VRMHumanBoneName.Head,
  VRMHumanBoneName.LeftShoulder,
  VRMHumanBoneName.RightShoulder,
  VRMHumanBoneName.LeftUpperArm,
  VRMHumanBoneName.LeftLowerArm,
  VRMHumanBoneName.LeftHand,
  VRMHumanBoneName.RightUpperArm,
  VRMHumanBoneName.RightLowerArm,
  VRMHumanBoneName.RightHand,
  VRMHumanBoneName.LeftUpperLeg,
  VRMHumanBoneName.LeftLowerLeg,
  VRMHumanBoneName.LeftFoot,
  VRMHumanBoneName.LeftToes,
  VRMHumanBoneName.RightUpperLeg,
  VRMHumanBoneName.RightLowerLeg,
  VRMHumanBoneName.RightFoot,
  VRMHumanBoneName.RightToes,
]

const tempQuat = new Quaternion()
const tempQuat2 = new Quaternion()
const tempVec = new Vector3()
const tempVec2 = new Vector3()

const Avatar: React.FC = () => {
  const { camera } = useThree()
  const [gltf, setGltf] = useState<GLTF>()
  const avatar = useRef<VRM | null>(null)
  const hipsRawRef = useRef<Object3D | null>(null)
  const boneMapRef = useRef<Partial<Record<VRMHumanBoneName, Object3D>>>({})
  const restQuatRef = useRef<Partial<Record<VRMHumanBoneName, Quaternion>>>({})

  const [prompt, setPrompt] = useState('')
  const [speechText, setSpeechText] = useState('')
  const [statusMessage, setStatusMessage] = useState('Enter a prompt to begin')
  const [segments, setSegments] = useState<SegmentState[]>([])
  const [currentSegmentIdx, setCurrentSegmentIdx] = useState(0)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)

  const segmentsRef = useRef<SegmentState[]>([])
  const currentSegmentRef = useRef<number>(0)
  const frameIdxRef = useRef<number>(0)
  const lastUpdateRef = useRef<number>(0)
  const rootOffsetRef = useRef<Vector3>(new Vector3())
  const lastRootRef = useRef<Vector3>(new Vector3())
  const loadedUrlsRef = useRef<Set<string>>(new Set())
  const isPlayingRef = useRef(false)

  useEffect(() => {
    segmentsRef.current = segments
  }, [segments])

  useEffect(() => {
    currentSegmentRef.current = currentSegmentIdx
  }, [currentSegmentIdx])

  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

  const controls = useControls('Facial Controls', {
    Neutral: { value: 0, min: 0, max: 1 },
    Angry: { value: 0, min: 0, max: 1 },
    Relaxed: { value: 0, min: 0, max: 1 },
    Happy: { value: 0, min: 0, max: 1 },
    Sad: { value: 0, min: 0, max: 1 },
    Surprised: { value: 0, min: 0, max: 1 },
    Extra: { value: 0, min: 0, max: 1 },
    LookDown: { value: 0, min: 0, max: 1 },
  })

  useEffect(() => {
    if (gltf) return
    const loader = new GLTFLoader()
    loader.register((parser) => new VRMLoaderPlugin(parser))

    loader.load(
      '/modelA.vrm',
      (loaded) => {
        const vrm: VRM = loaded.userData.vrm
        avatar.current = vrm
        vrm.lookAt.target = camera

        const hipsNormalized = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Hips)
        if (hipsNormalized) {
          hipsNormalized.rotation.y = Math.PI
        }
        hipsRawRef.current = vrm.humanoid.getRawBoneNode(VRMHumanBoneName.Hips) ?? hipsNormalized ?? null

        const boneMap: Partial<Record<VRMHumanBoneName, Object3D>> = {}
        TRACKED_BONES.forEach((boneName) => {
          const node = vrm.humanoid.getNormalizedBoneNode(boneName)
          if (node) {
            boneMap[boneName] = node
            // Cache rest-pose local rotation so we can apply motion relative to it
            restQuatRef.current[boneName] = node.quaternion.clone()
          }
        })
        boneMapRef.current = boneMap
        setGltf(loaded)
      },
      undefined,
      (error) => {
        const message = error instanceof Error ? error.message : String(error)
        setStatusMessage(`Failed to load VRM: ${message}`)
      }
    )
  }, [gltf, camera])

  useEffect(() => {
    segments.forEach((segment) => {
      if (!segment.motionData && !loadedUrlsRef.current.has(segment.motionUrl)) {
        loadedUrlsRef.current.add(segment.motionUrl)
        fetch(segment.motionUrl)
          .then((res) => {
            if (!res.ok) throw new Error(res.statusText)
            return res.json()
          })
          .then((data: MotionData) => {
            setSegments((prev) => prev.map((item) => (item.motionUrl === segment.motionUrl ? { ...item, motionData: data } : item)))
          })
          .catch((err) => {
            setStatusMessage(`Failed to load motion: ${err.message}`)
          })
      }
    })
  }, [segments])

  const applyFrame = useCallback((frame: MotionFrame, motion: MotionData) => {
    const root = frame.root_position ?? [0, 0, 0]
    lastRootRef.current.set(root[0], root[1], root[2])
    const hipsNode = hipsRawRef.current
    if (hipsNode) {
      tempVec.set(root[0], root[1], root[2]).add(rootOffsetRef.current)
      hipsNode.position.copy(tempVec)
    }

    const rots = frame.rotations ?? []
    const joints = motion.joints
    for (let i = 0; i < joints.length && i < rots.length; i += 1) {
      const boneName = HML_TO_VRM[joints[i]]
      if (!boneName) continue
      const bone = boneMapRef.current[boneName]
      if (!bone) continue
      const quat = rots[i]
      if (!quat || quat.length < 4) continue
      // Convert from [w,x,y,z] to THREE order [x,y,z,w]
      tempQuat.set(quat[1], quat[2], quat[3], quat[0])

      // Apply relative to the VRM rest-pose local rotation to account for rig pre-rotations
      const rest = restQuatRef.current[boneName]
      if (rest) {
        tempQuat2.copy(rest).multiply(tempQuat) // new = rest * motion
        bone.quaternion.copy(tempQuat2)
      } else {
        bone.quaternion.copy(tempQuat)
      }
    }
  }, [])

  useEffect(() => {
    if (segments.length === 0) return
    const ready = segments.every((segment) => !!segment.motionData)
    if (!ready) return

    rootOffsetRef.current.set(0, 0, 0)
    lastRootRef.current.set(0, 0, 0)
    frameIdxRef.current = 0
    lastUpdateRef.current = 0
    setCurrentSegmentIdx(0)
    setIsPlaying(true)
    setStatusMessage('Playing motion')

    const firstSegment = segments[0]
    if (firstSegment.motionData) {
      applyFrame(firstSegment.motionData.frames[0], firstSegment.motionData)
    }
  }, [segments, applyFrame])

  useFrame(({ clock }, delta) => {
    const vrm = avatar.current
    if (vrm) {
      vrm.update(delta)
    }

    if (!isPlayingRef.current) {
      if (vrm) {
        vrm.expressionManager.setValue('neutral', controls.Neutral)
        vrm.expressionManager.setValue('angry', controls.Angry)
        vrm.expressionManager.setValue('relaxed', controls.Relaxed)
        vrm.expressionManager.setValue('happy', controls.Happy)
        vrm.expressionManager.setValue('sad', controls.Sad)
        vrm.expressionManager.setValue('Surprised', controls.Surprised)
        vrm.expressionManager.setValue('Extra', controls.Extra)
        vrm.expressionManager.setValue('lookleft', controls.LookDown)
      }
      return
    }

    const segmentList = segmentsRef.current
    if (segmentList.length === 0) return
    const index = currentSegmentRef.current
    const activeSegment = segmentList[index]
    if (!activeSegment || !activeSegment.motionData) return

    const motion = activeSegment.motionData
    const frames = motion.frames
    if (!frames || frames.length === 0) return

    const now = clock.getElapsedTime()
    const step = 1 / (motion.fps || 20)

    if (now - lastUpdateRef.current >= step) {
      lastUpdateRef.current = now
      frameIdxRef.current += 1
      if (frameIdxRef.current >= frames.length) {
        const nextIdx = index + 1
        if (nextIdx < segmentList.length && segmentList[nextIdx].motionData) {
          tempVec.copy(rootOffsetRef.current).add(lastRootRef.current)
          const nextFirst = segmentList[nextIdx].motionData!.frames[0].root_position ?? [0, 0, 0]
          tempVec2.set(nextFirst[0], nextFirst[1], nextFirst[2])
          rootOffsetRef.current.copy(tempVec.sub(tempVec2))
          frameIdxRef.current = 0
          currentSegmentRef.current = nextIdx
          setCurrentSegmentIdx(nextIdx)
          setStatusMessage(`Playing segment ${nextIdx + 1}/${segmentList.length}`)

          const nextFrame = segmentList[nextIdx].motionData!.frames[0]
          applyFrame(nextFrame, segmentList[nextIdx].motionData!)
          return
        }
        isPlayingRef.current = false
        setIsPlaying(false)
        setStatusMessage('Playback finished')
        return
      }
    }

    const frame = frames[Math.min(frameIdxRef.current, frames.length - 1)]
    applyFrame(frame, motion)
  })

  const submitPrompt = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!prompt.trim()) return

    try {
      setIsGenerating(true)
      setIsPlaying(false)
      isPlayingRef.current = false
      setSegments([])
      loadedUrlsRef.current.clear()
      frameIdxRef.current = 0
      lastUpdateRef.current = 0
      rootOffsetRef.current.set(0, 0, 0)
      lastRootRef.current.set(0, 0, 0)
      setStatusMessage('Planning motion...')

      const response = await fetch(`${API_BASE}/api/conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || response.statusText)
      }
      const data = await response.json() as { speech_text: string, segments: { description: string, duration_seconds: number, motion_url: string }[] }
      setSpeechText(data.speech_text)
      setSegments(data.segments.map((segment) => ({
        description: segment.description,
        duration_seconds: segment.duration_seconds,
        motionUrl: segment.motion_url,
      })))
      setCurrentSegmentIdx(0)
      setStatusMessage('Loading motion clips...')
    } catch (err: any) {
      setStatusMessage(`Error: ${err.message ?? err}`)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <>
      {gltf ? <primitive object={gltf.scene} /> : <Html center>Loading avatar...</Html>}
      <Html fullscreen wrapperClass="ui-wrapper">
        <div className="ui-panel">
          <form className="prompt-form" onSubmit={submitPrompt}>
            <label htmlFor="prompt">Conversation prompt</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.currentTarget.value)}
              placeholder="Describe what the character should say and how they should move"
              rows={3}
            />
            <button type="submit" disabled={isGenerating || !prompt.trim()}>
              {isGenerating ? 'Generating...' : 'Generate motion'}
            </button>
          </form>

          <div className="status">{statusMessage}</div>

          {speechText && (
            <div className="speech-block">
              <h3>Speech</h3>
              <p>{speechText}</p>
            </div>
          )}

          {segments.length > 0 && (
            <div className="segments">
              <h4>Movement plan</h4>
              <ol>
                {segments.map((segment, index) => (
                  <li key={segment.motionUrl} className={index === currentSegmentIdx ? 'active' : ''}>
                    <div className="segment-desc">{segment.description}</div>
                    <div className="segment-meta">~{segment.duration_seconds.toFixed(1)}s · {segment.motionData ? 'ready' : 'loading...'}</div>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </Html>
    </>
  )
}

const Experience: React.FC = () => (
  <Canvas camera={{ position: [0, 1.3, 0.6] }}>
    <ambientLight intensity={0.65} />
    <spotLight position={[0, 2, -1]} intensity={0.4} />
    <Suspense fallback={null}>
      <Avatar />
    </Suspense>
    <OrbitControls target={[0, 1.3, 0]} />
  </Canvas>
)

const container = document.getElementById('root')
if (!container) throw new Error('Root container not found')
const root = createRoot(container)
root.render(<Experience />)
