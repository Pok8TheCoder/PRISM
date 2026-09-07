import { Crosshair, Rocket, Power, PowerOff, FlaskConical, Disc, TerminalSquare } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ScriptDef } from '../../types/session'

const iconFor = (id: string): LucideIcon => {
  if (id.startsWith('killchain')) return Crosshair
  if (id === 'auto-attack') return Rocket
  if (id === 'lab-up') return Power
  if (id === 'lab-down') return PowerOff
  if (id === 'bench-fair-ids') return FlaskConical
  if (id === 'forecast-record') return Disc
  return TerminalSquare
}

interface ScriptLauncherProps {
  scripts: ScriptDef[]
  onRun: (id: string) => void
  running?: string | null
}

export function ScriptLauncher({ scripts, onRun, running }: ScriptLauncherProps) {
  return (
    <div className="border-t border-[var(--border)] p-3">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
        Script launcher
      </p>
      <div className="grid grid-cols-2 gap-1.5">
        {scripts.map((s) => {
          const Icon = iconFor(s.id)
          const isRunning = running === s.id
          return (
            <button
              key={s.id}
              type="button"
              title={s.description}
              disabled={isRunning}
              onClick={() => onRun(s.id)}
              className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] px-2 py-1.5 text-left text-[11px] font-medium text-[var(--text-secondary)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text-primary)] disabled:opacity-50"
            >
              <Icon size={13} className={isRunning ? 'animate-spin' : ''} style={{ color: 'var(--accent)' }} />
              <span className="truncate">{isRunning ? 'Running…' : s.label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
