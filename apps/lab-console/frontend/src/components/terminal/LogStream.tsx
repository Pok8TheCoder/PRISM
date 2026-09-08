import { useEffect, useRef } from 'react'
import clsx from 'clsx'
import { Info, GitBranch, AlertTriangle, ShieldX, Activity, Crosshair } from 'lucide-react'
import type { LogLine } from '../../types/session'

const kindMeta: Record<LogLine['kind'], { color: string; icon: typeof Info }> = {
  info: { color: 'var(--text-muted)', icon: Info },
  phase: { color: 'var(--accent-2)', icon: GitBranch },
  alert: { color: 'var(--badge-warn)', icon: AlertTriangle },
  block: { color: 'var(--badge-error)', icon: ShieldX },
  scorer: { color: 'var(--badge-ok)', icon: Activity },
  attack: { color: 'var(--badge-error)', icon: Crosshair },
}

const fallbackMeta = kindMeta.info

interface LogStreamProps {
  lines: LogLine[]
  autoScroll: boolean
}

export function LogStream({ lines, autoScroll }: LogStreamProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!autoScroll || !scrollRef.current) return
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [lines, autoScroll])

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--terminal-bg)]">
      <div
        ref={scrollRef}
        className="flex-1 space-y-0.5 overflow-y-auto p-2 font-mono text-[11.5px] leading-relaxed"
      >
        {lines.length === 0 ? (
          <p className="px-1 py-2 text-[11px] text-[var(--text-muted)]">Waiting for lab events…</p>
        ) : (
          lines.map((line, i) => {
            const meta = kindMeta[line.kind] ?? fallbackMeta
            const Icon = meta.icon
            return (
              <div
                key={`${line.ts}-${i}`}
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
          })
        )}
      </div>
    </div>
  )
}
