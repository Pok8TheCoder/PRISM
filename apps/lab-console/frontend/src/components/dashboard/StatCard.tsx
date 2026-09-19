import clsx from 'clsx'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { IconTile } from '../common/IconTile'

interface StatCardProps {
  title: string
  icon?: LucideIcon
  tone?: 'brand' | 'ok' | 'warn' | 'error' | 'info'
  children: ReactNode
  className?: string
  action?: ReactNode
}

export function StatCard({
  title,
  icon,
  tone = 'brand',
  children,
  className,
  action,
}: StatCardProps) {
  return (
    <div
      className={clsx(
        'glass-panel animate-fade-up rounded-xl p-4 md:p-5 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[var(--shadow-glow)]',
        className,
      )}
    >
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          {icon && (
            <IconTile
              icon={icon}
              tone={tone}
              size={32}
            />
          )}

          <h3 className="truncate text-[13px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            {title}
          </h3>
        </div>

        {action}
      </div>

      <div className="min-w-0">
        {children}
      </div>
    </div>
  )
}