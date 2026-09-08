import { useCallback, useEffect, useRef, useState } from 'react'



export interface LivePlaybackOptions {

  /** Latest scored timeline position (window index × window_sec). */

  streamSec: number

  /** Scorer capture window length in seconds (default 1). */

  windowSec?: number

  enabled: boolean

}



/**

 * Live playhead extrapolates between scorer window polls so the head moves

 * smoothly even when each scored window takes several wall-clock seconds.

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

        rateRef.current = Math.min(Math.max(measured, 0.05), 4)

      }

    }

    lastSampleRef.current = { sec: streamSec, at: now }

    anchorRef.current = { sec: streamSec, at: now }

    targetRef.current = streamSec

  }, [streamSec, enabled])



  useEffect(() => {

    if (!enabled) return

    const now = performance.now()

    anchorRef.current = { sec: streamSec, at: now }

    lastSampleRef.current = { sec: streamSec, at: now }

    rateRef.current = 1

    viewRef.current = streamSec

    targetRef.current = streamSec

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

        const elapsedWall = (performance.now() - at) / 1000

        const extrapolated = sec + elapsedWall * rateRef.current

        const cap = targetRef.current + windowSec * 0.98

        const head = Math.min(extrapolated, cap)

        const diff = head - viewRef.current

        if (Math.abs(diff) > 0.004) {

          viewRef.current += diff * 0.35

        } else {

          viewRef.current = head

        }

      }

      setViewSec(viewRef.current)

      raf = requestAnimationFrame(tick)

    }

    raf = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(raf)

  }, [enabled, windowSec])



  const toggle = useCallback(() => {

    setPlaying((p) => {

      const next = !p

      playingRef.current = next

      if (next) {

        const { sec, at } = anchorRef.current

        const elapsedWall = (performance.now() - at) / 1000

        const head = Math.min(sec + elapsedWall * rateRef.current, targetRef.current + windowSec * 0.98)

        viewRef.current = head

        setViewSec(head)

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

    const elapsedWall = (performance.now() - at) / 1000

    viewRef.current = Math.min(sec + elapsedWall * rateRef.current, targetRef.current + windowSec * 0.98)

    setViewSec(viewRef.current)

    setPlaying(true)

  }, [windowSec])



  const seek = useCallback(

    (sec: number) => {

      const cap = targetRef.current + windowSec * 0.98

      const clamped = Math.max(0, Math.min(cap, sec))

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



  const liveEdgeSec = Math.min(

    targetRef.current + windowSec * 0.98,

    anchorRef.current.sec + ((performance.now() - anchorRef.current.at) / 1000) * rateRef.current,

  )

  const offsetSec = viewSec - streamSec



  return {

    viewSec: enabled ? viewSec : streamSec,

    liveEdgeSec: enabled ? Math.max(streamSec, liveEdgeSec) : streamSec,

    offsetSec: enabled ? offsetSec : 0,

    playing: enabled ? playing : true,

    toggle,

    play,

    pause,

    seek,

    skip,

  }

}


