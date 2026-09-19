import clsx from 'clsx'
import type { ReactNode } from 'react'

export type BadgeTone =
  | 'ok'
  | 'warn'
  | 'error'
  | 'info'
  | 'neutral'
  | 'brand'

const toneStyles: Record<BadgeTone, string> = {
  ok: 'bg-[var(--badge-ok-soft)] text-[var(--badge-ok)]',
  warn: 'bg-[var(--badge-warn-soft)] text-[var(--badge-warn)]',
  error: 'bg-[var(--badge-error-soft)] text-[var(--badge-error)]',
  info: 'bg-[var(--badge-info-soft)] text-[var(--badge-info)]',
  neutral: 'bg-[var(--bg-elevated)] text-[var(--text-muted)]',
  brand: 'bg-[var(--accent-soft)] text-[var(--accent)]',
}

interface BadgeProps {
  tone?: BadgeTone
  children: ReactNode
  dot?: boolean
  className?: string
}

export function Badge({
  tone = 'neutral',
  children,
  dot,
  className,
}: BadgeProps) {
  return (
    <span
      className={clsx(
        'chip inline-flex items-center gap-1.5 whitespace-nowrap',
        toneStyles[tone],
        className,
      )}
    >
      {dot && (
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ background: 'currentColor' }}
        />
      )}

      {children}
    </span>
  )
}

export function LiveDot({
  tone = 'ok' as BadgeTone,
}: {
  tone?: BadgeTone
}) {
  const color =
    tone === 'ok'
      ? 'var(--badge-ok)'
      : tone === 'warn'
        ? 'var(--badge-warn)'
        : tone === 'error'
          ? 'var(--badge-error)'
          : 'var(--badge-info)'

  return (
    <span className="relative inline-flex h-2 w-2">
      <span
        className="absolute inline-flex h-full w-full rounded-full opacity-75"
        style={{
          background: color,
          animation: 'pulse-dot 1.6s ease-in-out infinite',
        }}
      />

      <span
        className="relative inline-flex h-2 w-2 rounded-full"
        style={{ background: color }}
      />
    </span>
  )
}