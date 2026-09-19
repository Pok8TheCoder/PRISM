import { Bell } from 'lucide-react'
import { LiveDot } from '../components/common/Badge'

interface TopBarProps {
  title: string
  subtitle?: string
  right?: React.ReactNode
}

export function TopBar({
  title,
  subtitle,
  right,
}: TopBarProps) {
  return (
    <header className="flex min-h-[72px] items-center justify-between border-b border-[var(--border)] bg-[var(--bg-surface)]/70 px-5 py-3 backdrop-blur-xl lg:px-6">
      {/* Left side */}
      <div className="min-w-0">
        <div className="flex items-center gap-2.5">
          <h1 className="truncate text-[18px] font-bold tracking-tight text-[var(--text-primary)]">
            {title}
          </h1>

          <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--bg-elevated)] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--text-muted)]">
            <LiveDot tone="ok" />
            live
          </span>
        </div>

        {subtitle && (
          <p className="mt-1 truncate text-xs text-[var(--text-muted)]">
            {subtitle}
          </p>
        )}
      </div>

      {/* Right side */}
      <div className="ml-4 flex shrink-0 items-center gap-2.5">
        {right}

        <button
          type="button"
          aria-label="Notifications"
          className="group flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-muted)] transition-all duration-200 hover:border-[var(--border-strong)] hover:bg-[var(--bg-glass)] hover:text-[var(--text-primary)]"
        >
          <Bell
            size={16}
            strokeWidth={2}
            className="transition-transform duration-200 group-hover:scale-105"
          />
        </button>
      </div>
    </header>
  )
}