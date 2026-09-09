import { useState } from 'react'
import clsx from 'clsx'
import { ChevronLeft, ChevronRight, ScrollText, TerminalSquare, Rows3, Columns2, PanelRight } from 'lucide-react'
import type { LayoutMode, LogLine, PolicyMode, ScriptDef } from '../../types/session'
import { LogStream } from './LogStream'
import { MockTerminal } from './MockTerminal'
import { ScriptLauncher } from './ScriptLauncher'
import { Badge } from '../common/Badge'

interface TerminalRailProps {
  logs: LogLine[]
  scripts: ScriptDef[]
  layout: LayoutMode
  policyMode: PolicyMode
  collapsed: boolean
  onCollapse: () => void
  onLayoutChange: (layout: LayoutMode) => void
  onRunScript: (id: string) => void
}

const layoutOptions: { mode: LayoutMode; icon: typeof Rows3; title: string }[] = [
  { mode: 'charts-only', icon: Rows3, title: 'Charts only' },
  { mode: 'split', icon: Columns2, title: 'Split' },
  { mode: 'terminal-focus', icon: PanelRight, title: 'Terminal focus' },
]

export function TerminalRail({
  logs,
  scripts,
  layout,
  policyMode,
  collapsed,
  onCollapse,
  onLayoutChange,
  onRunScript,
}: TerminalRailProps) {
  const [tab, setTab] = useState<'log' | 'shell'>('log')
  const [running, setRunning] = useState<string | null>(null)

  const handleRun = (id: string) => {
    setRunning(id)
    onRunScript(id)
    setTimeout(() => setRunning(null), 1500)
  }

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={onCollapse}
        className="flex w-9 shrink-0 flex-col items-center gap-2 border-l border-[var(--border)] bg-[var(--bg-surface)]/70 py-4 text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
        title="Expand terminal"
      >
        <ChevronLeft size={15} />
        <TerminalSquare size={15} />
      </button>
    )
  }

  return (
    <aside className="flex w-[340px] min-w-[300px] max-w-[420px] shrink-0 flex-col border-l border-[var(--border)] bg-[var(--bg-surface)]/70 backdrop-blur-xl">
      <div className="flex items-center justify-between border-b border-[var(--border)] px-3 py-2.5">
        <span className="flex items-center gap-1.5 text-sm font-semibold text-[var(--text-primary)]">
          <TerminalSquare size={15} style={{ color: 'var(--accent)' }} />
          Terminal
        </span>
        <div className="flex items-center gap-1">
          <div className="flex gap-0.5 rounded-lg bg-[var(--bg-elevated)] p-0.5">
            {layoutOptions.map(({ mode, icon: Icon, title }) => (
              <button
                key={mode}
                type="button"
                title={title}
                onClick={() => onLayoutChange(mode)}
                className={clsx(
                  'flex h-6 w-6 items-center justify-center rounded-md transition-colors',
                  layout === mode ? 'text-white' : 'text-[var(--text-muted)]',
                )}
                style={layout === mode ? { background: 'var(--gradient-brand)' } : undefined}
              >
                <Icon size={13} />
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onCollapse}
            className="flex h-6 w-6 items-center justify-center rounded-md text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-[11px] text-[var(--text-muted)]">Policy</span>
        <Badge tone={policyMode === 'ips' ? 'error' : 'info'} dot>
          {policyMode.toUpperCase()}
        </Badge>
      </div>

      <div className="mx-3 mb-2 flex gap-1 rounded-lg bg-[var(--bg-elevated)] p-1">
        <button
          type="button"
          className={clsx(
            'flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-colors',
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
            'flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-colors',
            tab === 'shell' ? 'text-white' : 'text-[var(--text-muted)]',
          )}
          style={tab === 'shell' ? { background: 'var(--gradient-brand)' } : undefined}
          onClick={() => setTab('shell')}
        >
          <TerminalSquare size={13} /> Shell
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden px-3">
        {tab === 'log' ? <LogStream lines={logs} /> : <MockTerminal />}
      </div>
      <ScriptLauncher scripts={scripts} onRun={handleRun} running={running} />
    </aside>
  )
}
