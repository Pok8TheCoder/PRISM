const lineItems: { color: string; label: string }[] = [
  { color: 'var(--chart-actual)', label: 'Actual network' },
  { color: 'var(--chart-predicted)', label: 'Model forecast' },
]

const attackItems: { color: string; label: string }[] = [
  { color: '#3B82F6', label: 'Recon' },
  { color: '#A855F7', label: 'Enumeration' },
  { color: '#EF4444', label: 'Brute Force' },
  { color: '#F59E0B', label: 'Suspicious' },
]

const detectionItems: {
  style: 'dashed' | 'solid' | 'strong'
  label: string
}[] = [
  { style: 'dashed', label: 'Predicted' },
  { style: 'solid', label: 'Actual' },
  { style: 'strong', label: 'Both' },
]

function LineLegendItem({
  color,
  label,
}: {
  color: string
  label: string
}) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{
          background: color,
          boxShadow: `0 0 5px ${color}`,
        }}
      />
      {label}
    </span>
  )
}

function AttackLegendItem({
  color,
  label,
}: {
  color: string
  label: string
}) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{
          background: color,
          boxShadow: `0 0 5px ${color}`,
        }}
      />
      {label}
    </span>
  )
}

function DetectionLegendItem({
  style,
  label,
}: {
  style: 'dashed' | 'solid' | 'strong'
  label: string
}) {
  const borderStyle =
    style === 'dashed'
      ? 'dashed'
      : 'solid'

  const borderWidth =
    style === 'strong'
      ? 3
      : 2

  const opacity =
    style === 'dashed'
      ? 0.65
      : style === 'solid'
        ? 0.9
        : 1

  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        className="inline-block w-5"
        style={{
          borderTop: `${borderWidth}px ${borderStyle} var(--text-secondary)`,
          opacity,
        }}
      />
      {label}
    </span>
  )
}

export function ChartLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)]/60 px-3 py-2.5 text-[11px] text-[var(--text-muted)]">
      {/* Lines */}
      <div className="flex items-center gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
          Lines
        </span>

        {lineItems.map((item) => (
          <LineLegendItem
            key={item.label}
            color={item.color}
            label={item.label}
          />
        ))}
      </div>

      <span className="hidden h-4 w-px bg-[var(--border)] sm:block" />

      {/* Attack types */}
      <div className="flex items-center gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
          Attack type
        </span>

        {attackItems.map((item) => (
          <AttackLegendItem
            key={item.label}
            color={item.color}
            label={item.label}
          />
        ))}
      </div>

      <span className="hidden h-4 w-px bg-[var(--border)] sm:block" />

      {/* Detection state */}
      <div className="flex items-center gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
          Detection
        </span>

        {detectionItems.map((item) => (
          <DetectionLegendItem
            key={item.label}
            style={item.style}
            label={item.label}
          />
        ))}
      </div>
    </div>
  )
}