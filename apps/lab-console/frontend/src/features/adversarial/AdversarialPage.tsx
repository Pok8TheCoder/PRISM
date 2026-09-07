import { useState } from 'react'
import { BrainCircuit, X, Layers, Cpu, Tag as TagIcon, Calendar } from 'lucide-react'
import { mockModelRegistry } from '../../mocks/models'
import type { ModelRegistryEntry } from '../../types/models'
import { TopBar } from '../../app/TopBar'
import { IconTile } from '../../components/common/IconTile'
import { Badge } from '../../components/common/Badge'

export function AdversarialPage() {
  const [selected, setSelected] = useState<ModelRegistryEntry | null>(null)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar title="Adversarial Lab" subtitle="Model registry — agent-ready metadata for versions & checkpoints" />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto p-6">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {mockModelRegistry.map((m, i) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setSelected(m)}
                className="glass-panel animate-fade-up rounded-2xl p-5 text-left transition-all duration-300 hover:-translate-y-1 hover:shadow-[var(--shadow-glow)]"
                style={{ animationDelay: `${i * 40}ms` }}
              >
                <div className="mb-3 flex items-center justify-between">
                  <IconTile icon={BrainCircuit} tone="brand" size={38} />
                  <Badge tone="brand">{m.version}</Badge>
                </div>
                <h3 className="text-base font-semibold text-[var(--text-primary)]">{m.name}</h3>
                <p className="mt-0.5 truncate font-mono text-[11px] text-[var(--text-muted)]">{m.checkpointPath}</p>

                <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-lg bg-[var(--bg-elevated)] py-2">
                    <p className="text-sm font-bold text-[var(--text-primary)]">{m.sizeMb}</p>
                    <p className="text-[9px] uppercase text-[var(--text-muted)]">MB</p>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-elevated)] py-2">
                    <p className="text-sm font-bold text-[var(--text-primary)]">{m.featureDim}</p>
                    <p className="text-[9px] uppercase text-[var(--text-muted)]">Features</p>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-elevated)] py-2">
                    <p className="text-sm font-bold text-[var(--text-primary)]">{m.classes.length}</p>
                    <p className="text-[9px] uppercase text-[var(--text-muted)]">Classes</p>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-1">
                  {m.tags.map((t) => (
                    <Badge key={t} tone="neutral">
                      {t}
                    </Badge>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </div>

        {selected && (
          <aside className="glass-panel w-[380px] shrink-0 overflow-auto border-l p-5">
            <div className="mb-4 flex items-start justify-between">
              <div className="flex items-center gap-2.5">
                <IconTile icon={BrainCircuit} tone="brand" size={36} />
                <div>
                  <h3 className="font-semibold text-[var(--text-primary)]">{selected.name}</h3>
                  <p className="text-xs text-[var(--text-muted)]">{selected.version}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="flex h-7 w-7 items-center justify-center rounded-lg text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
              >
                <X size={15} />
              </button>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2 rounded-xl bg-[var(--bg-elevated)] p-3">
                <Cpu size={15} style={{ color: 'var(--accent)' }} />
                <div>
                  <p className="text-[10px] uppercase text-[var(--text-muted)]">Parameters</p>
                  <p className="text-sm font-semibold">{selected.parameters.toLocaleString()}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-xl bg-[var(--bg-elevated)] p-3">
                <Layers size={15} style={{ color: 'var(--accent-2)' }} />
                <div>
                  <p className="text-[10px] uppercase text-[var(--text-muted)]">Feature dim / seq len</p>
                  <p className="text-sm font-semibold">
                    {selected.featureDim}d · seq {selected.seqLen}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-xl bg-[var(--bg-elevated)] p-3">
                <Calendar size={15} style={{ color: 'var(--badge-ok)' }} />
                <div>
                  <p className="text-[10px] uppercase text-[var(--text-muted)]">Last trained</p>
                  <p className="text-sm font-semibold">{selected.lastTrained ?? '—'}</p>
                </div>
              </div>
              <div className="rounded-xl bg-[var(--bg-elevated)] p-3">
                <p className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase text-[var(--text-muted)]">
                  <TagIcon size={12} /> Classes ({selected.classes.length})
                </p>
                <p className="max-h-28 overflow-y-auto font-mono text-[11px] leading-relaxed text-[var(--text-secondary)]">
                  {selected.classes.join(', ')}
                </p>
              </div>
              <div>
                <p className="mb-1 text-[10px] uppercase text-[var(--text-muted)]">Raw metadata (agent export)</p>
                <pre className="max-h-64 overflow-auto rounded-xl bg-[var(--terminal-bg)] p-3 text-[10px] leading-relaxed text-[var(--terminal-fg)]">
                  {JSON.stringify(selected, null, 2)}
                </pre>
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  )
}
