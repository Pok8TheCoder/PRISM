import { useCallback, useEffect, useRef, useState } from 'react'

export interface PlaybackClockOptions {
  durationSec: number
  initialSec?: number
  live?: boolean
  speed?: number
}

export function usePlaybackClock({
  durationSec,
  initialSec = 0,
  live = false,
  speed = 1,
}: PlaybackClockOptions) {
  const [playing, setPlaying] = useState(live)
  const [playheadSec, setPlayheadSec] = useState(initialSec)
  const [playbackSpeed, setPlaybackSpeed] = useState(speed)
  const lastFrameRef = useRef<number | null>(null)
  const playheadRef = useRef(playheadSec)
  playheadRef.current = playheadSec

  useEffect(() => {
    setPlayheadSec(initialSec)
  }, [initialSec])

  useEffect(() => {
    if (!playing) {
      lastFrameRef.current = null
      return
    }

    let raf = 0
    const tick = (now: number) => {
      if (lastFrameRef.current == null) lastFrameRef.current = now
      const dt = (now - lastFrameRef.current) / 1000
      lastFrameRef.current = now
      const next = Math.min(durationSec, playheadRef.current + dt * playbackSpeed)
      playheadRef.current = next
      setPlayheadSec(next)
      if (next >= durationSec && !live) {
        setPlaying(false)
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, playbackSpeed, durationSec, live])

  const play = useCallback(() => setPlaying(true), [])
  const pause = useCallback(() => setPlaying(false), [])
  const toggle = useCallback(() => setPlaying((p) => !p), [])

  const seek = useCallback(
    (sec: number) => {
      const clamped = Math.max(0, Math.min(durationSec, sec))
      playheadRef.current = clamped
      setPlayheadSec(clamped)
    },
    [durationSec],
  )

  const skip = useCallback(
    (delta: number) => seek(playheadRef.current + delta),
    [seek],
  )

  return {
    playheadSec,
    playing,
    playbackSpeed,
    setPlaybackSpeed,
    play,
    pause,
    toggle,
    seek,
    skip,
  }
}
