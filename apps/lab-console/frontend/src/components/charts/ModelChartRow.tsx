import { BrainCircuit } from 'lucide-react'
import type { ModelSlot, SessionState } from '../../types/session'
import { TimeSeriesChart } from './TimeSeriesChart'
import { Badge } from '../common/Badge'
import { IconTile } from '../common/IconTile'

interface ModelChartRowProps {
  model: ModelSlot
  session: SessionState
  playheadSec: number
}

function getModelIndex(modelName: string): string {
  const normalized = modelName.toLowerCase()

  if (normalized.includes('shaun')) {
    return '01'
  }

  if (normalized.includes('hx')) {
    return '02'
  }

  return '—'
}

export function ModelChartRow({
  model,
  session,
  playheadSec,
}: ModelChartRowProps) {
  const activeRegion = model.regions.find(
    (r) =>
      r.start <= playheadSec &&
      playheadSec <= r.end,
  )

  const modelIndex = getModelIndex(model.name)

  return (
    <section className="animate-fade-up mb-7">
      {/* Model identity / telemetry header */}
      <div className="mb-2.5 flex flex-wrap items-end justify-between gap-3 border-b border-[var(--border)] pb-2.5">
        {/* Left: model identity */}
        <div className="flex min-w-0 items-center gap-3">
          {/* Model number */}
          <div className="flex h-7 w-7 shrink-0 items-center justify-center border border-[var(--border)] bg-[var(--bg-surface)] text-[9px] font-bold tracking-[0.08em] text-[var(--text-muted)]">
            {modelIndex}
          </div>

          <IconTile
            icon={BrainCircuit}
            tone="brand"
            size={30}
          />

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
              <h3 className="truncate text-[15px] font-bold tracking-tight text-[var(--text-primary)]">
                {model.name}
              </h3>

              <span className="text-[8px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
                Forecast model
              </span>
            </div>

            <p className="mt-0.5 text-[10px] text-[var(--text-muted)]">
              Network attack probability · live scoring track
            </p>
          </div>
        </div>

        {/* Right: model telemetry */}
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          {activeRegion ? (
            <Badge
              tone={
                activeRegion.kind === 'attack'
                  ? 'error'
                  : 'warn'
              }
              dot
            >
              {activeRegion.label}
            </Badge>
          ) : (
            <Badge tone="ok" dot>
              clear
            </Badge>
          )}

          <div className="hidden h-4 w-px bg-[var(--border)] sm:block" />

          <Badge tone="brand">
            MAE {model.accuracy.lineMae.toFixed(3)}
          </Badge>

          <Badge tone="info">
            P{' '}
            {(
              model.accuracy.regionPrecision *
              100
            ).toFixed(0)}
            %
          </Badge>

          <Badge tone="info">
            R{' '}
            {(
              model.accuracy.regionRecall *
              100
            ).toFixed(0)}
            %
          </Badge>
        </div>
      </div>

      {/* Analysis surface */}
      <div className="overflow-hidden border border-[var(--border)] bg-[var(--bg-surface-2)]">
        <TimeSeriesChart
          actual={session.actual}
          observed={model.observed}
          predicted={model.predicted}
          modelRegions={model.regions}
          groundTruthRegions={
            session.groundTruthRegions
          }
          memoryRegions={
            session.memoryRegions
          }
          playheadSec={playheadSec}
          windowSec={26}
          dataStepSec={
            session.windowSec ?? 1
          }
          accuracy={model.accuracy}
          height={200}
        />
      </div>
    </section>
  )
}