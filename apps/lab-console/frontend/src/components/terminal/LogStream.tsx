import clsx from 'clsx'
import { Info, GitBranch, AlertTriangle, ShieldX, Activity } from 'lucide-react'
import type { LogLine } from '../../types/session'

const kindMeta: Record<LogLine['kind'], { color: string; icon: typeof Info }> = {
  info: { color: 'var(--text-muted)', icon: Info },
  phase: { color: 'var(--accent-2)', icon: GitBranch },
  alert: { color: 'var(--badge-warn)', icon: AlertTriangle },
  block: { color: 'var(--badge-error)', icon: ShieldX },
  scorer: { color: 'var(--badge-ok)', icon: Activity },
}

interface LogStreamProps {
  lines: LogLine[]
}

export function LogStream({ lines }: LogStreamProps) {
  return (
    <div className="flex-1 space-y-0.5 overflow-y-auto pr-1 font-mono text-[11.5px] leading-relaxed">
      {lines.map((line, i) => {
        const meta = kindMeta[line.kind]
        const Icon = meta.icon
        return (
          <div
            key={i}
            className={clsx(
              'flex items-start gap-1.5 rounded-md px-1.5 py-1 animate-fade-up',
              i === lines.length - 1 && 'bg-[var(--bg-elevated)]',
            )}
          >
            <Icon size={12} className="mt-0.5 shrink-0" color={meta.color} />
            <span className="shrink-0 opacity-50">{line.ts.toFixed(0)}s</span>
            <span style={{ color: meta.color }}>{line.text}</span>
          </div>
        )
      })}
    </div>
  )
}
