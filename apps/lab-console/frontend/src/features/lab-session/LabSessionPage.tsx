import { useCallback, useEffect, useState } from 'react'
import clsx from 'clsx'
import {
  Radio,
  FileVideo,
  Eye,
  ShieldBan,
  Circle,
  Disc,
} from 'lucide-react'

import { ModelChartRow } from '../../components/charts/ModelChartRow'
import { ChartLegend } from '../../components/charts/ChartLegend'
import { PlaybackBar } from '../../components/charts/PlaybackBar'
import { TerminalRail } from '../../components/terminal/TerminalRail'
import { TerminalDock } from '../../components/terminal/TerminalDock'
import { Badge, LiveDot } from '../../components/common/Badge'
import { ApiSourceBadge } from '../../components/common/ApiSourceBadge'
import { TopBar } from '../../app/TopBar'

import { usePlaybackClock } from '../../hooks/usePlaybackClock'
import { useLivePlayback } from '../../hooks/useLivePlayback'
import {
  useLogStream,
  useLabConfig,
  useScripts,
  useSessionData,
} from '../../hooks/useLabApi'

import type {
  LayoutMode,
  SessionMode,
} from '../../types/session'

const SESSION_ID = 'default'

export function LabSessionPage() {
  const [sourceMode, setSourceMode] =
    useState<SessionMode>('recorded')

  const [layout, setLayout] =
    useState<LayoutMode>('split')

  const [railCollapsed, setRailCollapsed] =
    useState(false)

  const [recording, setRecording] =
    useState(false)

  const [selectedModels, setSelectedModels] =
    useState<string[]>([])

  const {
    session,
    source,
    setPolicy,
  } = useSessionData(
    SESSION_ID,
    sourceMode,
  )

  const logs = useLogStream()

  const {
    scripts,
    launch,
  } = useScripts()

  const {
    config: labConfig,
    update: updateLabConfig,
  } = useLabConfig()

  const isLive =
    sourceMode === 'live'

  /*
   * Select every model by default.
   *
   * Each model remains an independent graph.
   * Shaun v3 and HX-C are NOT combined.
   */
  useEffect(() => {
    if (
      selectedModels.length === 0 &&
      session.models.length > 0
    ) {
      setSelectedModels(
        session.models.map(
          (model) => model.id,
        ),
      )
    }
  }, [
    session.models,
    selectedModels.length,
  ])

  /*
   * Recorded playback.
   */
  const clock = usePlaybackClock({
    durationSec:
      session.durationSec,
    initialSec:
      session.playheadSec,
    live: false,
    speed: 1,
  })

  /*
   * Live playback.
   */
  const livePlayback =
    useLivePlayback({
      streamSec:
        session.playheadSec,
      windowSec:
        session.windowSec ?? 1,
      enabled: isLive,
    })

  const playheadSec = isLive
    ? livePlayback.viewSec
    : clock.playheadSec

  const durationSec = isLive
    ? livePlayback.liveEdgeSec
    : session.durationSec

  /*
   * Model selector.
   */
  const toggleModel = (
    modelId: string,
  ) => {
    setSelectedModels(
      (current) => {
        if (
          current.includes(modelId)
        ) {
          return current.filter(
            (id) => id !== modelId,
          )
        }

        return [
          ...current,
          modelId,
        ]
      },
    )
  }

  /*
   * Run a script.
   */
  const handleRunScript =
    useCallback(
      (
        id: string,
        delaySec?: number,
      ) => {
        launch(id, delaySec)
      },
      [launch],
    )

  /*
   * Start / stop recording.
   */
  const handleRecord =
    useCallback(() => {
      if (recording) {
        setRecording(false)
        return
      }

      setRecording(true)
      launch('forecast-record')
    }, [
      recording,
      launch,
    ])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">

      {/* =====================================================
          TOP BAR
          ===================================================== */}

      <TopBar
        title="Lab Session"
        subtitle="Multi-model playback, IDS/IPS testing & scripts"
        right={
          <ApiSourceBadge
            source={source}
          />
        }
      />

      {/* =====================================================
          SESSION CONTROL BAR
          ===================================================== */}

      <div className="shrink-0 px-4 pt-3">
        <div className="flex min-h-[58px] flex-wrap items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2">

          {/* SESSION / SOURCE */}
          <div className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] px-1.5 py-1">
            <span className="px-1.5 text-[9px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Source
            </span>

            <div className="flex rounded-md bg-[var(--bg-elevated)] p-0.5">
              <button
                type="button"
                onClick={() => setSourceMode('live')}
                className={clsx(
                  'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] font-medium transition-colors',
                  sourceMode === 'live'
                    ? 'bg-[var(--badge-ok-soft)] text-[var(--badge-ok)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]',
                )}
              >
                <Radio size={12} />
                Live
              </button>

              <button
                type="button"
                onClick={() => setSourceMode('recorded')}
                className={clsx(
                  'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] font-medium transition-colors',
                  sourceMode === 'recorded'
                    ? 'bg-[var(--badge-info-soft)] text-[var(--badge-info)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--badge-info)]',
                )}
              >
                <FileVideo size={12} />
                Recorded
              </button>
            </div>
          </div>

          {/* DETECTION POLICY */}
          <div className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] px-1.5 py-1">
            <span className="px-1.5 text-[9px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Policy
            </span>

            <div className="flex rounded-md bg-[var(--bg-elevated)] p-0.5">
              <button
                type="button"
                onClick={() => setPolicy('ids')}
                className={clsx(
                  'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] font-medium transition-colors',
                  session.policyMode === 'ids'
                    ? 'bg-[var(--badge-info-soft)] text-[var(--badge-info)]'
                    : 'text-[var(--text-muted)] hover:bg-[var(--badge-info-soft)] hover:text-[var(--badge-info)]',
                )}
              >
                <Eye size={12} />
                IDS
              </button>

              <button
                type="button"
                onClick={() => setPolicy('ips')}
                className={clsx(
                  'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] font-medium transition-colors',
                  session.policyMode === 'ips'
                    ? 'bg-[var(--badge-error-soft)] text-[var(--badge-error)]'
                    : 'text-[var(--text-muted)] hover:bg-[var(--badge-error-soft)] hover:text-[var(--badge-error)]',
                )}
              >
                <ShieldBan size={12} />
                IPS
              </button>
            </div>

            {session.policyMode === 'ips' && (
              <Badge
                tone={session.ips.armed ? 'warn' : 'neutral'}
                dot
              >
                {session.ips.armed ? 'armed' : 'disarmed'}
              </Badge>
            )}
          </div>

          {/* MODELS */}
          <div className="flex min-w-0 items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] px-1.5 py-1">
            <span className="px-1.5 text-[9px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Models
            </span>

            <div className="flex flex-wrap gap-1">
              {session.models.map((model) => {
                const active = selectedModels.includes(model.id)

                return (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => toggleModel(model.id)}
                    className={clsx(
                      'rounded-md border px-2.5 py-1.5 text-[11px] font-medium transition-all',
                      active
                        ? model.id.toLowerCase().includes('shaun')
                          ? 'border-[rgba(168,85,247,0.35)] bg-[rgba(168,85,247,0.10)] text-[#A855F7]'
                          : model.id.toLowerCase().includes('hx')
                            ? 'border-[rgba(34,197,94,0.35)] bg-[rgba(34,197,94,0.10)] text-[#22C55E]'
                            : 'border-[var(--border-strong)] bg-[var(--accent-soft)] text-[var(--text-primary)]'
                        : 'border-transparent text-[var(--text-muted)] hover:border-[var(--border)] hover:text-[var(--text-primary)]',
                    )}
                  >
                    {model.name}
                  </button>
                )
              })}
            </div>
          </div>

          {/* SESSION ACTIONS */}
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {isLive && (
              <span className="flex items-center gap-1.5 rounded-md border border-[var(--badge-ok-soft)] bg-[var(--badge-ok-soft)] px-2 py-1.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--badge-ok)]">
                <LiveDot tone="ok" />
                Live
              </span>
            )}

            <button
              type="button"
              onClick={handleRecord}
              className={clsx(
                'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] font-semibold transition-colors',
                recording
                  ? 'bg-[var(--badge-error)] text-white shadow-sm'
                  : 'border border-[var(--badge-error-soft)] bg-[var(--badge-error-soft)] text-[var(--badge-error)] hover:border-[var(--badge-error)] hover:bg-[var(--badge-error-soft)]',
              )}
            >
              {recording ? (
                <Circle size={10} fill="currentColor" />
              ) : (
                <Disc size={13} />
              )}

              {recording ? 'Stop' : 'Record'}
            </button>
          </div>
        </div>
      </div>

      {/* =====================================================
          MAIN WORKSPACE
          ===================================================== */}

      <div className="flex min-h-0 flex-1 gap-3 px-4 pb-0 pt-3">

        {/* ===================================================
            MODEL ANALYSIS
            =================================================== */}

        <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">

          {/* SECTION TITLE */}

          <div className="mb-2 flex shrink-0 items-end justify-between border-b border-[var(--border)] px-1 pb-2">

            <div>
              <div className="flex items-center gap-2">
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
                  Model analysis
                </p>

                <span className="text-[9px] text-[var(--text-muted)]">
                  /
                </span>

                <span className="text-[9px] font-medium uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                  Forecast evaluation
                </span>
              </div>

              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                Forecast probability and network events
              </p>
            </div>

            {isLive && (
              <span className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-[var(--badge-ok)]">
                <LiveDot tone="ok" />
                streaming
              </span>
            )}

          </div>

          {/* LEGEND */}

          <div className="mb-2 shrink-0">
            <ChartLegend />
          </div>

          {/* =================================================
              MODEL GRAPHS

              Each model gets its own complete graph.

              Shaun v3
              ─────────────
              complete graph

              HX-C
              ─────────────
              complete graph

              They are NOT merged.
              ================================================= */}

          <div className="min-h-0 flex-1 overflow-y-auto pr-1">

            <div className="space-y-1 pb-2">

              {session.models
                .filter((model) =>
                  selectedModels.includes(
                    model.id,
                  ),
                )
                .map((model) => (
                  <ModelChartRow
                    key={model.id}
                    model={model}
                    session={session}
                    playheadSec={
                      playheadSec
                    }
                  />
                ))}

              {/* EMPTY STATE */}

              {session.models.filter(
                (model) =>
                  selectedModels.includes(
                    model.id,
                  ),
              ).length === 0 && (
                <div className="flex min-h-[240px] items-center justify-center border border-dashed border-[var(--border)] bg-[var(--bg-surface)]">

                  <div className="text-center">

                    <p className="text-sm font-medium text-[var(--text-secondary)]">
                      No models selected
                    </p>

                    <p className="mt-1 text-xs text-[var(--text-muted)]">
                      Select Shaun v3, HX-C, or another model above.
                    </p>

                  </div>
                </div>
              )}

            </div>
          </div>

          {/* =================================================
              PLAYBACK
              ================================================= */}

          <div className="shrink-0 border-t border-[var(--border)] pb-1 pt-2">

            <PlaybackBar
              playheadSec={
                playheadSec
              }

              durationSec={
                durationSec
              }

              playing={
                isLive
                  ? livePlayback.playing
                  : clock.playing
              }

              speed={
                clock.playbackSpeed
              }

              timeMode={
                isLive
                  ? 'live'
                  : 'recorded'
              }

              offsetSec={
                isLive
                  ? livePlayback.offsetSec
                  : undefined
              }

              onToggle={
                isLive
                  ? livePlayback.toggle
                  : clock.toggle
              }

              onSeek={
                isLive
                  ? livePlayback.seek
                  : clock.seek
              }

              onSkip={
                isLive
                  ? livePlayback.skip
                  : clock.skip
              }

              onSpeedChange={
                clock.setPlaybackSpeed
              }
            />

          </div>

        </main>

        {/* ===================================================
            SCRIPT / CONFIG RAIL
            =================================================== */}

        <aside
          className={clsx(
            'hidden min-h-0 shrink-0 lg:flex',
            railCollapsed
              ? 'w-12'
              : 'w-[290px] xl:w-[310px]',
          )}
        >

          <TerminalRail
            scripts={scripts}
            layout={layout}
            policyMode={
              session.policyMode
            }
            collapsed={
              railCollapsed
            }
            onCollapse={() =>
              setRailCollapsed(
                (value) =>
                  !value,
              )
            }
            onLayoutChange={
              setLayout
            }
            onRunScript={
              handleRunScript
            }
            labConfig={
              labConfig
            }
            onLabConfigChange={
              updateLabConfig
            }
          />

        </aside>

      </div>

      {/* =====================================================
          OUTPUT / LOG
          ===================================================== */}

      <div className="shrink-0 px-4 pb-3 pt-2">

        <div className="h-[130px] overflow-hidden rounded-xl border border-[var(--border)]">

          <TerminalDock
            logs={logs}
          />

        </div>

      </div>

    </div>
  )
}