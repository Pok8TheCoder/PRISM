import { useEffect, useState } from 'react'
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

const KILLCHAIN_IDS = new Set([
  'killchain-recon',
  'killchain-enum',
  'killchain-spray',
  'killchain-loot',
  'killchain-all',
])

export interface LabConfigState {
  horizonSec: number
  attackDelaySec: number | null
}

interface ScriptLauncherProps {
  scripts: ScriptDef[]
  labConfig: LabConfigState
  onLabConfigChange: (patch: Partial<LabConfigState>) => void
  onRun: (id: string, delaySec?: number) => void
  running?: string | null
}

export function ScriptLauncher({ scripts, labConfig, onLabConfigChange, onRun, running }: ScriptLauncherProps) {
  const [attackDelayInput, setAttackDelayInput] = useState(
    labConfig.attackDelaySec != null ? String(labConfig.attackDelaySec) : '',
  )

  useEffect(() => {
    setAttackDelayInput(labConfig.attackDelaySec != null ? String(labConfig.attackDelaySec) : '')
  }, [labConfig.attackDelaySec])

  const parseAttackDelay = (): number | undefined => {
    const trimmed = attackDelayInput.trim()
    if (!trimmed) return undefined
    const n = Number(trimmed)
    return Number.isFinite(n) && n >= 0 ? n : undefined
  }

  const handleRun = (id: string) => {
    if (KILLCHAIN_IDS.has(id)) {
      onRun(id, parseAttackDelay())
      return
    }
    onRun(id)
  }

  return (
    <div className="p-3">
      <div className="mb-3 space-y-1">
        <label className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Forecast horizon (s)
        </label>
        <input
          type="number"
          min={10}
          max={600}
          step={5}
          value={labConfig.horizonSec}
          onChange={(e) => onLabConfigChange({ horizonSec: Number(e.target.value) || 60 })}
          className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] px-2 py-1.5 text-[11px] text-[var(--text-primary)]"
          title="How far ahead each model forecasts (seconds)"
        />
        <p className="text-[10px] text-[var(--text-muted)]">Default 60s — applies on next scorer window.</p>
      </div>

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
              onClick={() => handleRun(s.id)}
              className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] px-2 py-1.5 text-left text-[11px] font-medium text-[var(--text-secondary)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text-primary)] disabled:opacity-50"
            >
              <Icon size={13} className={isRunning ? 'animate-spin' : ''} style={{ color: 'var(--accent)' }} />
              <span className="truncate">{isRunning ? 'Running…' : s.label}</span>
            </button>
          )
        })}
      </div>

      <div className="mt-3 space-y-1">
        <label className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          Attack delay (s)
        </label>
        <input
          type="number"
          min={0}
          step={1}
          placeholder="30–60 random if empty"
          value={attackDelayInput}
          onChange={(e) => setAttackDelayInput(e.target.value)}
          onBlur={() => {
            const n = parseAttackDelay()
            onLabConfigChange({ attackDelaySec: n ?? null })
          }}
          className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] px-2 py-1.5 text-[11px] text-[var(--text-primary)]"
          title="Kill-chain attacks fire after this many seconds (empty = random 30–60s)"
        />
      </div>
    </div>
  )
}
