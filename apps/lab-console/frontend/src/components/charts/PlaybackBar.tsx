import clsx from 'clsx'
import { Play, Pause, Rewind, FastForward, SkipBack } from 'lucide-react'

export type PlaybackTimeMode = 'recorded' | 'live'

interface PlaybackBarProps {
  playheadSec: number
  durationSec: number
  playing: boolean
  speed: number
  timeMode?: PlaybackTimeMode
  /** viewSec − liveEdge; negative when behind the live stream. */
  offsetSec?: number
  onToggle: () => void
  onSeek: (sec: number) => void
  onSkip: (delta: number) => void
  onSpeedChange: (speed: number) => void
}

const SPEEDS = [0.5, 1, 2, 4, 8]

function fmtHms(sec: number): string {
  const abs = Math.abs(sec)
  const h = Math.floor(abs / 3600)
  const m = Math.floor((abs % 3600) / 60)
  const s = Math.floor(abs % 60)
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

function fmtSigned(sec: number): string {
  if (Math.abs(sec) < 0.05) return '+0:00'
  const sign = sec < 0 ? '−' : '+'
  return `${sign}${fmtHms(sec)}`
}

function fmtRecorded(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function PlaybackBar({
  playheadSec,
  durationSec,
  playing,
  speed,
  timeMode = 'recorded',
  offsetSec = 0,
  onToggle,
  onSeek,
  onSkip,
  onSpeedChange,
}: PlaybackBarProps) {
  const live = timeMode === 'live'
  const max = Math.max(durationSec, playheadSec, 0.1)
  const pct = max > 0 ? (playheadSec / max) * 100 : 0

  return (
    <div className="border-t border-[var(--border)] bg-[var(--bg-surface)]/70 px-4 py-3 backdrop-blur-xl">
      <div className="mb-2.5 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => onSeek(0)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
          title="Jump to session start"
        >
          <SkipBack size={15} />
        </button>
        <button
          type="button"
          onClick={() => onSkip(-5)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
          title="-5s"
        >
          <Rewind size={15} />
        </button>
        <button
          type="button"
          onClick={onToggle}
          className="flex h-9 w-9 items-center justify-center rounded-xl text-white shadow-lg transition-transform hover:-translate-y-0.5"
          style={{ background: 'var(--gradient-brand)', boxShadow: '0 4px 16px -4px var(--accent-glow)' }}
          title={live ? (playing ? 'Pause view (stream continues)' : 'Catch up to live') : undefined}
        >
          {playing ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" className="ml-0.5" />}
        </button>
        <button
          type="button"
          onClick={() => onSkip(5)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
          title="+5s"
        >
          <FastForward size={15} />
        </button>

        <span className="ml-1 font-mono text-xs font-medium text-[var(--text-secondary)]">
          {live ? (
            <>
              <span className={offsetSec < -0.5 ? 'text-[var(--badge-warn)]' : undefined}>
                {fmtSigned(offsetSec)}
              </span>
              <span className="text-[var(--text-muted)]"> · </span>
              <span>+{fmtHms(playheadSec)}</span>
              <span className="text-[var(--text-muted)]"> / +{fmtHms(durationSec)}</span>
            </>
          ) : (
            <>
              {fmtRecorded(playheadSec)} <span className="text-[var(--text-muted)]">/ {fmtRecorded(durationSec)}</span>
            </>
          )}
        </span>

        {!live && (
          <div className="ml-auto flex gap-1 rounded-lg bg-[var(--bg-elevated)] p-1">
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onSpeedChange(s)}
                className={clsx(
                  'rounded-md px-2 py-0.5 text-xs font-semibold transition-colors',
                  speed === s ? 'text-white' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]',
                )}
                style={speed === s ? { background: 'var(--gradient-brand)' } : undefined}
              >
                {s}x
              </button>
            ))}
          </div>
        )}

        {live && (
          <span className="ml-auto text-[10px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
            {playing ? 'following live' : 'paused · stream continues'}
          </span>
        )}
      </div>
      <div className="relative h-1.5 w-full rounded-full bg-[var(--bg-elevated)]">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-75 ease-linear"
          style={{ width: `${pct}%`, background: 'var(--gradient-brand)' }}
        />
        <input
          type="range"
          min={0}
          max={max}
          step={0.05}
          value={playheadSec}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="absolute inset-0 h-1.5 w-full cursor-pointer appearance-none bg-transparent"
        />
      </div>
      {live && (
        <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--text-muted)]">
          <span>+0:00 start</span>
          <span>+{fmtHms(durationSec)} live</span>
        </div>
      )}
    </div>
  )
}
