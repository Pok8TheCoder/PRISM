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

export function StatCard({ title, icon, tone = 'brand', children, className, action }: StatCardProps) {
  return (
    <div
      className={clsx(
        'glass-panel animate-fade-up rounded-2xl p-5 transition-shadow duration-300 hover:shadow-[var(--shadow-glow)]',
        className,
      )}
    >
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          {icon && <IconTile icon={icon} tone={tone} size={32} />}
          <h3 className="text-[13px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">{title}</h3>
        </div>
        {action}
      </div>
      {children}
    </div>
  )
}
