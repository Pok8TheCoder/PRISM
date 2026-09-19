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

export function MetricTile({
  label,
  value,
  delta,
  deltaTone = 'ok',
  icon,
  tone = 'brand',
}: MetricTileProps) {
  const deltaColor =
    deltaTone === 'ok'
      ? 'var(--badge-ok)'
      : deltaTone === 'warn'
        ? 'var(--badge-warn)'
        : 'var(--badge-error)'

  return (
    <div className="group relative overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-surface)]/80 p-4 transition-all duration-200 hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]">
      {/* Subtle top accent */}
      <div
        className="absolute inset-x-0 top-0 h-px opacity-0 transition-opacity duration-200 group-hover:opacity-100"
        style={{ background: 'var(--gradient-brand)' }}
      />

      <div className="flex items-center gap-3.5">
        <IconTile
          icon={icon}
          tone={tone}
          size={44}
        />

        <div className="min-w-0 flex-1">
          <p className="truncate text-[10px] font-semibold uppercase tracking-[0.1em] text-[var(--text-muted)]">
            {label}
          </p>

          <div className="mt-1.5 flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="shrink-0 text-[27px] font-bold leading-none tracking-tight text-[var(--text-primary)]">
              {value}
            </span>

            {delta && (
              <span
                className="min-w-0 text-[11px] font-semibold leading-tight"
                style={{ color: deltaColor }}
              >
                {delta}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}