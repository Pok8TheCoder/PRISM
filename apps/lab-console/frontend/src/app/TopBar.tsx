import { Bell } from 'lucide-react'
import { LiveDot } from '../components/common/Badge'

interface TopBarProps {
  title: string
  subtitle?: string
  right?: React.ReactNode
}

export function TopBar({ title, subtitle, right }: TopBarProps) {
  return (
    <div className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--bg-surface)]/50 px-6 py-4 backdrop-blur-xl">
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">{title}</h1>
          <span className="flex items-center gap-1.5 rounded-full bg-[var(--bg-elevated)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            <LiveDot tone="ok" /> live
          </span>
        </div>
        {subtitle && <p className="mt-0.5 text-xs text-[var(--text-muted)]">{subtitle}</p>}
      </div>
      <div className="flex items-center gap-3">
        {right}
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
        >
          <Bell size={16} />
        </button>
      </div>
    </div>
  )
}
