import type { LucideIcon } from 'lucide-react'

interface IconTileProps {
  icon: LucideIcon
  tone?: 'brand' | 'ok' | 'warn' | 'error' | 'info'
  size?: number
}

const toneBg: Record<string, string> = {
  brand: 'var(--accent-soft)',
  ok: 'var(--badge-ok-soft)',
  warn: 'var(--badge-warn-soft)',
  error: 'var(--badge-error-soft)',
  info: 'var(--badge-info-soft)',
}

const toneColor: Record<string, string> = {
  brand: 'var(--accent)',
  ok: 'var(--badge-ok)',
  warn: 'var(--badge-warn)',
  error: 'var(--badge-error)',
  info: 'var(--badge-info)',
}

export function IconTile({ icon: Icon, tone = 'brand', size = 36 }: IconTileProps) {
  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-xl border"
      style={{
        width: size,
        height: size,
        background: toneBg[tone],
        borderColor: tone === 'brand' ? 'var(--border)' : 'transparent',
      }}
    >
      <Icon size={size * 0.52} color={toneColor[tone]} strokeWidth={2.25} />
    </div>
  )
}
