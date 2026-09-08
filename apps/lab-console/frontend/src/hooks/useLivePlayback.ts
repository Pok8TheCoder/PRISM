import { useCallback, useEffect, useRef, useState } from 'react'

export interface LivePlaybackOptions {
  /** Scored timeline position (window index × window length). */
  streamSec: number
  /** Scorer capture window length in seconds (default 1). */
  windowSec?: number
  enabled: boolean
}

/**
 * Smooth live playhead on the scored timeline (not wall-clock elapsed).
 * Measures scorer cadence and extrapolates between window polls.
 */
export function useLivePlayback({ streamSec, windowSec = 1, enabled }: LivePlaybackOptions) {
  const [viewSec, setViewSec] = useState(streamSec)
  const [playing, setPlaying] = useState(true)

  const viewRef = useRef(viewSec)
  const playingRef = useRef(playing)
  const targetRef = useRef(streamSec)
  const anchorRef = useRef({ sec: streamSec, at: performance.now() })
  const rateRef = useRef(1)
  const lastSampleRef = useRef({ sec: streamSec, at: performance.now() })

  viewRef.current = viewSec
  playingRef.current = playing

  useEffect(() => {
    if (!enabled) return
    const now = performance.now()
    const prev = lastSampleRef.current
    if (streamSec > prev.sec + 0.001) {
      const dtWall = (now - prev.at) / 1000
      if (dtWall > 0.05) {
        const measured = (streamSec - prev.sec) / dtWall
        rateRef.current = Math.min(Math.max(measured, 0.05), 2)
      }
    }
    lastSampleRef.current = { sec: streamSec, at: now }
    targetRef.current = streamSec
    const projected = anchorRef.current.sec + ((now - anchorRef.current.at) / 1000) * rateRef.current
    const err = streamSec - projected
    if (Math.abs(err) > windowSec * 1.5) {
      anchorRef.current = { sec: streamSec, at: now }
      viewRef.current = streamSec
    }
  }, [streamSec, enabled, windowSec])

  useEffect(() => {
    if (!enabled) return
    const now = performance.now()
    anchorRef.current = { sec: streamSec, at: now }
    targetRef.current = streamSec
    lastSampleRef.current = { sec: streamSec, at: now }
    rateRef.current = 1
    viewRef.current = streamSec
    setViewSec(streamSec)
    setPlaying(true)
    playingRef.current = true
  }, [enabled]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!enabled) return

    let raf = 0
    const tick = () => {
      if (playingRef.current) {
        const { sec, at } = anchorRef.current
        const head = sec + ((performance.now() - at) / 1000) * rateRef.current
        const cap = targetRef.current + windowSec * 0.98
        const clamped = Math.min(head, cap)
        const diff = clamped - viewRef.current
        if (Math.abs(diff) > 0.003) {
          viewRef.current += diff * 0.4
        } else {
          viewRef.current = clamped
        }
      }
      setViewSec(viewRef.current)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [enabled, windowSec])

  const scoredEdge = () => targetRef.current + windowSec * 0.98

  const toggle = useCallback(() => {
    setPlaying((p) => {
      const next = !p
      playingRef.current = next
      if (next) {
        const { sec, at } = anchorRef.current
        viewRef.current = Math.min(sec + ((performance.now() - at) / 1000) * rateRef.current, scoredEdge())
        setViewSec(viewRef.current)
      }
      return next
    })
  }, [windowSec])

  const pause = useCallback(() => {
    playingRef.current = false
    setPlaying(false)
  }, [])

  const play = useCallback(() => {
    playingRef.current = true
    const { sec, at } = anchorRef.current
    viewRef.current = Math.min(sec + ((performance.now() - at) / 1000) * rateRef.current, scoredEdge())
    setViewSec(viewRef.current)
    setPlaying(true)
  }, [windowSec])

  const seek = useCallback(
    (sec: number) => {
      const clamped = Math.max(0, Math.min(scoredEdge(), sec))
      viewRef.current = clamped
      setViewSec(clamped)
    },
    [windowSec],
  )

  const skip = useCallback(
    (delta: number) => {
      seek(viewRef.current + delta)
    },
    [seek],
  )

  const liveEdgeSec = scoredEdge()

  return {
    viewSec: enabled ? viewSec : streamSec,
    liveEdgeSec: enabled ? Math.max(streamSec, liveEdgeSec) : streamSec,
    offsetSec: enabled ? viewSec - streamSec : 0,
    playing: enabled ? playing : true,
    toggle,
    play,
    pause,
    seek,
    skip,
  }
}
