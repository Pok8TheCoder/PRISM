import { useState } from 'react'
import clsx from 'clsx'
import { ScrollText, TerminalSquare } from 'lucide-react'
import type { LogLine } from '../../types/session'
import { LogStream } from './LogStream'
import { LabTerminal } from './LabTerminal'

interface TerminalDockProps {
  logs: LogLine[]
}

export function TerminalDock({ logs }: TerminalDockProps) {
  const [tab, setTab] = useState<'log' | 'shell'>('log')
  const [autoScroll, setAutoScroll] = useState(true)

  return (
    <section className="flex h-[280px] shrink-0 flex-col overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)]/80 backdrop-blur-xl">
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border)] px-3 py-2">
        <span className="text-xs font-semibold text-[var(--text-primary)]">Output</span>
        <div className="flex gap-1 rounded-lg bg-[var(--bg-elevated)] p-0.5">
          <button
            type="button"
            className={clsx(
              'flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium transition-colors',
              tab === 'log' ? 'text-white' : 'text-[var(--text-muted)]',
            )}
            style={tab === 'log' ? { background: 'var(--gradient-brand)' } : undefined}
            onClick={() => setTab('log')}
          >
            <ScrollText size={13} /> Log
          </button>
          <button
            type="button"
            className={clsx(
              'flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium transition-colors',
              tab === 'shell' ? 'text-white' : 'text-[var(--text-muted)]',
            )}
            style={tab === 'shell' ? { background: 'var(--gradient-brand)' } : undefined}
            onClick={() => setTab('shell')}
          >
            <TerminalSquare size={13} /> Shell
          </button>
        </div>
        <label className="ml-auto flex cursor-pointer items-center gap-1.5 text-[10px] text-[var(--text-muted)]">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-[var(--border)] accent-[var(--accent)]"
          />
          Autoscroll
        </label>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-2">
        <div className={tab === 'log' ? 'flex h-full min-h-0 flex-1 flex-col' : 'hidden'}>
          <LogStream lines={logs} autoScroll={autoScroll} />
        </div>
        <div className={tab === 'shell' ? 'flex h-full min-h-0 flex-1 flex-col' : 'hidden'}>
          <LabTerminal autoScroll={autoScroll} />
        </div>
      </div>
    </section>
  )
}
