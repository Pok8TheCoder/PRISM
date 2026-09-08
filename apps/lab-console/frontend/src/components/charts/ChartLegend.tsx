const items: { color: string; label: string }[] = [
  { color: 'var(--chart-predicted)', label: 'Model prediction' },
  { color: 'var(--chart-actual)', label: 'Actual network' },
  { color: 'var(--region-suspicious-border)', label: 'Suspicious' },
  { color: 'var(--region-attack-border)', label: 'Attack' },
  { color: 'var(--region-suspicious-resolved-border)', label: 'Resolved (was sus.)' },
  { color: 'var(--region-attack-resolved-border)', label: 'Resolved (was attack)' },
  { color: 'var(--region-gt-border)', label: 'Ground truth' },
  { color: 'var(--region-episodic-border)', label: 'RAMX episodic' },
  { color: 'var(--region-overlap-border)', label: 'Overlap' },
]

export function ChartLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)]/60 px-3 py-2 text-[11px] text-[var(--text-muted)]">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: item.color, boxShadow: `0 0 6px ${item.color}` }}
          />
          {item.label}
        </span>
      ))}
    </div>
  )
}
