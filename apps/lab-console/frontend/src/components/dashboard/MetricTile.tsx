import type { LucideIcon } from 'lucide-react'
import { IconTile } from '../common/IconTile'

interface MetricTileProps {
  label: string
  value: string
  delta?: string
  deltaTone?: 'ok' | 'warn' | 'error'
  icon: LucideIcon
  tone?: 'brand' | 'ok' | 'warn' | 'error' | 'info'
}

export function MetricTile({ label, value, delta, deltaTone = 'ok', icon, tone = 'brand' }: MetricTileProps) {
  const deltaColor =
    deltaTone === 'ok' ? 'var(--badge-ok)' : deltaTone === 'warn' ? 'var(--badge-warn)' : 'var(--badge-error)'

  return (
    <div className="glass-panel animate-fade-up flex items-center gap-4 rounded-2xl p-5 transition-transform duration-300 hover:-translate-y-0.5">
      <IconTile icon={icon} tone={tone} size={46} />
      <div className="min-w-0 flex-1">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">{label}</p>
        <div className="mt-0.5 flex items-baseline gap-2">
          <span className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">{value}</span>
          {delta && (
            <span className="text-xs font-semibold" style={{ color: deltaColor }}>
              {delta}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
