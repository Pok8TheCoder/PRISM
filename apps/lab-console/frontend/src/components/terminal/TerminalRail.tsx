import { useState } from 'react'
import clsx from 'clsx'
import { ChevronLeft, ChevronRight, TerminalSquare, Rows3, Columns2, PanelRight } from 'lucide-react'
import type { LayoutMode, PolicyMode, ScriptDef } from '../../types/session'
import { ScriptLauncher, type LabConfigState } from './ScriptLauncher'
import { Badge } from '../common/Badge'

interface TerminalRailProps {
  scripts: ScriptDef[]
  layout: LayoutMode
  policyMode: PolicyMode
  collapsed: boolean
  onCollapse: () => void
  onLayoutChange: (layout: LayoutMode) => void
  onRunScript: (id: string, delaySec?: number) => void
  labConfig: LabConfigState
  onLabConfigChange: (patch: Partial<LabConfigState>) => void
}

const layoutOptions: { mode: LayoutMode; icon: typeof Rows3; title: string }[] = [
  { mode: 'charts-only', icon: Rows3, title: 'Charts only' },
  { mode: 'split', icon: Columns2, title: 'Split' },
  { mode: 'terminal-focus', icon: PanelRight, title: 'Terminal focus' },
]

export function TerminalRail({
  scripts,
  layout,
  policyMode,
  collapsed,
  onCollapse,
  onLayoutChange,
  onRunScript,
  labConfig,
  onLabConfigChange,
}: TerminalRailProps) {
  const [running, setRunning] = useState<string | null>(null)

  const handleRun = (id: string, delaySec?: number) => {
    setRunning(id)
    onRunScript(id, delaySec)
    setTimeout(() => setRunning(null), 1500)
  }

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={onCollapse}
        className="flex w-9 shrink-0 flex-col items-center gap-2 border-l border-[var(--border)] bg-[var(--bg-surface)]/70 py-4 text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
        title="Expand scripts panel"
      >
        <ChevronLeft size={15} />
        <TerminalSquare size={15} />
      </button>
    )
  }

  return (
    <aside className="flex h-full min-h-0 w-[340px] min-w-[300px] max-w-[420px] shrink-0 flex-col border-l border-[var(--border)] bg-[var(--bg-surface)]/70 backdrop-blur-xl">
      <div className="flex shrink-0 items-center justify-between border-b border-[var(--border)] px-3 py-2.5">
        <span className="flex items-center gap-1.5 text-sm font-semibold text-[var(--text-primary)]">
          <TerminalSquare size={15} style={{ color: 'var(--accent)' }} />
          Scripts
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

      <div className="flex shrink-0 items-center justify-between border-b border-[var(--border)] px-3 py-2">
        <span className="text-[11px] text-[var(--text-muted)]">Policy</span>
        <Badge tone={policyMode === 'ips' ? 'error' : 'info'} dot>
          {policyMode.toUpperCase()}
        </Badge>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <ScriptLauncher
          scripts={scripts}
          labConfig={labConfig}
          onLabConfigChange={onLabConfigChange}
          onRun={handleRun}
          running={running}
        />
      </div>
    </aside>
  )
}
