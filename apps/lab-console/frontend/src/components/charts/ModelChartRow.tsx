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

export function ModelChartRow({ model, session, playheadSec }: ModelChartRowProps) {
  const activeRegion = model.regions.find((r) => r.start <= playheadSec && playheadSec <= r.end)

  return (
    <div className="glass-panel animate-fade-up mb-4 rounded-2xl p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-y-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <IconTile icon={BrainCircuit} tone="brand" size={30} />
          <div className="min-w-0">
            <h3 className="truncate text-sm font-semibold text-[var(--text-primary)]">{model.name}</h3>
            <p className="truncate text-[11px] text-[var(--text-muted)]">P(attack) forecast track</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          {activeRegion ? (
            <Badge tone={activeRegion.kind === 'attack' ? 'error' : 'warn'} dot>
              {activeRegion.label}
            </Badge>
          ) : (
            <Badge tone="ok" dot>
              clear
            </Badge>
          )}
          <Badge tone="brand">MAE {model.accuracy.lineMae.toFixed(3)}</Badge>
          <Badge tone="info">P {(model.accuracy.regionPrecision * 100).toFixed(0)}%</Badge>
          <Badge tone="info">R {(model.accuracy.regionRecall * 100).toFixed(0)}%</Badge>
        </div>
      </div>
      <TimeSeriesChart
        actual={session.actual}
        observed={model.observed}
        predicted={model.predicted}
        modelRegions={model.regions}
        groundTruthRegions={session.groundTruthRegions}
        memoryRegions={session.memoryRegions}
        playheadSec={playheadSec}
        windowSec={26}
        dataStepSec={session.windowSec ?? 1}
        accuracy={model.accuracy}
        height={200}
      />
    </div>
  )
}
