import { useCallback, useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { Radio, FileVideo, Eye, ShieldBan, Circle, Disc } from 'lucide-react'
import { ModelChartRow } from '../../components/charts/ModelChartRow'
import { ChartLegend } from '../../components/charts/ChartLegend'
import { PlaybackBar } from '../../components/charts/PlaybackBar'
import { TerminalRail } from '../../components/terminal/TerminalRail'
import { Badge, LiveDot } from '../../components/common/Badge'
import { ApiSourceBadge } from '../../components/common/ApiSourceBadge'
import { TopBar } from '../../app/TopBar'
import { usePlaybackClock } from '../../hooks/usePlaybackClock'
import { useLogStream, useScripts, useSessionData } from '../../hooks/useLabApi'
import type { LayoutMode, PolicyMode, SessionMode } from '../../types/session'

const SESSION_ID = 'default'

export function LabSessionPage() {
  const [sourceMode, setSourceMode] = useState<SessionMode>('recorded')
  const [layout, setLayout] = useState<LayoutMode>('split')
  const [railCollapsed, setRailCollapsed] = useState(false)
  const [recording, setRecording] = useState(false)
  const [selectedModels, setSelectedModels] = useState<string[]>([])

  const { session, source, setPolicy } = useSessionData(SESSION_ID, sourceMode)
  const logs = useLogStream()
  const { scripts, launch } = useScripts()

  const live = sourceMode === 'live'

  useEffect(() => {
    if (selectedModels.length === 0 && session.models.length > 0) {
      setSelectedModels(session.models.map((m) => m.id))
    }
  }, [session.models, selectedModels.length])

  const clock = usePlaybackClock({
    durationSec: session.durationSec,
    initialSec: session.playheadSec,
    live: false,
    speed: 1,
  })

  const playheadSec = live ? session.playheadSec : clock.playheadSec

  const policyMode: PolicyMode = session.policyMode

  const visibleModels = useMemo(
    () => session.models.filter((m) => selectedModels.includes(m.id)),
    [session.models, selectedModels],
  )

  const toggleModel = (id: string) => {
    setSelectedModels((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }

  const handleRunScript = useCallback(
    (id: string) => {
      launch(id)
    },
    [launch],
  )

  const handleRecord = useCallback(() => {
    if (!recording) {
      setRecording(true)
      launch('forecast-record')
      return
    }
    setRecording(false)
  }, [recording, launch])

  const showCharts = layout !== 'terminal-focus'
  const showRail = layout !== 'charts-only'

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar
        title="Lab Session"
        subtitle="Multi-model playback, IDS/IPS testing & scripts"
        right={<ApiSourceBadge source={source} />}
      />

      {/* Control toolbar */}
      <div className="mx-4 mt-4 flex flex-wrap items-center gap-3 rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)]/70 px-4 py-3 backdrop-blur-xl">
        <div className="flex gap-1 rounded-lg bg-[var(--bg-elevated)] p-1">
          <button
            type="button"
            onClick={() => setSourceMode('live')}
            className={clsx(
              'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors',
              sourceMode === 'live' ? 'text-white' : 'text-[var(--text-muted)]',
            )}
            style={sourceMode === 'live' ? { background: 'var(--gradient-brand)' } : undefined}
          >
            <Radio size={13} /> Live
          </button>
          <button
            type="button"
            onClick={() => setSourceMode('recorded')}
            className={clsx(
              'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors',
              sourceMode === 'recorded' ? 'text-white' : 'text-[var(--text-muted)]',
            )}
            style={sourceMode === 'recorded' ? { background: 'var(--gradient-brand)' } : undefined}
          >
            <FileVideo size={13} /> Recorded
          </button>
        </div>

        <div className="h-6 w-px bg-[var(--border)]" />

        <div className="flex gap-1 rounded-lg bg-[var(--bg-elevated)] p-1">
          <button
            type="button"
            onClick={() => setPolicy('ids')}
            className={clsx(
              'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-semibold uppercase transition-colors',
              policyMode === 'ids' ? 'text-white' : 'text-[var(--text-muted)]',
            )}
            style={policyMode === 'ids' ? { background: 'linear-gradient(135deg, #60a5fa, #3b82f6)' } : undefined}
          >
            <Eye size={13} /> IDS
          </button>
          <button
            type="button"
            onClick={() => setPolicy('ips')}
            className={clsx(
              'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-semibold uppercase transition-colors',
              policyMode === 'ips' ? 'text-white' : 'text-[var(--text-muted)]',
            )}
            style={policyMode === 'ips' ? { background: 'linear-gradient(135deg, #f87171, #ef4444)' } : undefined}
          >
            <ShieldBan size={13} /> IPS
          </button>
        </div>

        {policyMode === 'ips' && (
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={session.ips.armed ? 'warn' : 'neutral'} dot>
              {session.ips.armed ? 'armed' : 'disarmed'}
            </Badge>
            {session.ips.blocker && <Badge tone="error">blocked by {session.ips.blocker}</Badge>}
            {Object.entries(session.ips.streaks).map(([k, v]) => (
              <Badge key={k} tone="neutral">
                {k}: {v}
              </Badge>
            ))}
          </div>
        )}

        <div className="h-6 w-px bg-[var(--border)]" />

        <div className="flex flex-wrap gap-1.5">
          {session.models.map((m) => {
            const active = selectedModels.includes(m.id)
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => toggleModel(m.id)}
                className={clsx(
                  'rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors',
                  active
                    ? 'border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]'
                    : 'border-[var(--border)] text-[var(--text-muted)] opacity-60',
                )}
              >
                {m.name}
              </button>
            )
          })}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {live && (
            <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--badge-ok)]">
              <LiveDot tone="ok" /> streaming
            </span>
          )}
          <button
            type="button"
            onClick={handleRecord}
            className={clsx(
              'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors',
              recording ? 'text-white' : 'border border-[var(--border)] text-[var(--text-secondary)]',
            )}
            style={recording ? { background: 'var(--badge-error)' } : undefined}
          >
            {recording ? <Circle size={11} fill="white" /> : <Disc size={13} />}
            {recording ? 'Stop' : 'Record'}
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 gap-0 overflow-hidden p-4">
        {showCharts && (
          <div className="flex min-w-0 flex-1 flex-col gap-3">
            <ChartLegend />
            <div className="flex-1 overflow-y-auto pr-1">
              {visibleModels.map((model) => (
                <ModelChartRow key={model.id} model={model} session={session} playheadSec={playheadSec} />
              ))}
              {visibleModels.length === 0 && (
                <p className="p-6 text-center text-sm text-[var(--text-muted)]">Select at least one model.</p>
              )}
            </div>
            <PlaybackBar
              playheadSec={playheadSec}
              durationSec={session.durationSec}
              playing={live ? true : clock.playing}
              speed={clock.playbackSpeed}
              onToggle={live ? () => {} : clock.toggle}
              onSeek={live ? () => {} : clock.seek}
              onSkip={live ? () => {} : clock.skip}
              onSpeedChange={clock.setPlaybackSpeed}
            />
          </div>
        )}

        {showRail && (
          <div className="ml-4 flex">
            <TerminalRail
              logs={logs}
              scripts={scripts}
              layout={layout}
              policyMode={policyMode}
              collapsed={railCollapsed}
              onCollapse={() => setRailCollapsed((c) => !c)}
              onLayoutChange={setLayout}
              onRunScript={handleRunScript}
            />
          </div>
        )}
      </div>
    </div>
  )
}
