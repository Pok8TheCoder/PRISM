import {
  Boxes,
  Cpu,
  ShieldAlert,
  BrainCircuit,
  Database,
  Network,
  GraduationCap,
  Activity,
} from 'lucide-react'
import { StatCard } from '../../components/dashboard/StatCard'
import { MetricTile } from '../../components/dashboard/MetricTile'
import { AreaChart } from '../../components/dashboard/AreaChart'
import { Badge } from '../../components/common/Badge'
import { ApiSourceBadge } from '../../components/common/ApiSourceBadge'
import { TopBar } from '../../app/TopBar'
import { useDashboardData } from '../../hooks/useLabApi'

export function DashboardPage() {
  const { data: d, source } = useDashboardData()
  const runningCount = d.containers.filter((c) => c.status === 'running').length
  const loadedModels = d.models.filter((m) => m.loaded).length
  const latestP = d.metricsTimeseries[d.metricsTimeseries.length - 1]?.pAttack ?? 0

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <TopBar
        title="Dashboard"
        subtitle="Live network activity & background job insights"
        right={<ApiSourceBadge source={source} />}
      />
      <div className="flex-1 overflow-y-auto p-6">
        {/* KPI row */}
        <div className="mb-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricTile
            label="Containers running"
            value={`${runningCount}/${d.containers.length}`}
            delta="all healthy"
            deltaTone="ok"
            icon={Boxes}
            tone="brand"
          />
          <MetricTile
            label="P(attack) now"
            value={`${(latestP * 100).toFixed(0)}%`}
            delta={latestP > 0.5 ? 'elevated' : 'nominal'}
            deltaTone={latestP > 0.5 ? 'error' : 'ok'}
            icon={ShieldAlert}
            tone={latestP > 0.5 ? 'error' : 'ok'}
          />
          <MetricTile
            label="Models loaded"
            value={`${loadedModels}/${d.models.length}`}
            delta="HX-C · Shaun v3"
            deltaTone="ok"
            icon={BrainCircuit}
            tone="info"
          />
          <MetricTile
            label="GPU"
            value={d.device.cuda ? `${d.device.vramGb} GB` : 'CPU only'}
            delta={d.device.deviceName}
            deltaTone="ok"
            icon={Cpu}
            tone="brand"
          />
        </div>

        <div className="mb-5 grid gap-4 xl:grid-cols-3">
          <StatCard title="Network activity" icon={Activity} tone="brand" className="xl:col-span-2">
            <AreaChart data={d.metricsTimeseries} height={220} />
          </StatCard>

          <StatCard title="Protected networks" icon={Network} tone="info">
            <ul className="space-y-2.5">
              {d.networks.map((n) => (
                <li key={n.name} className="flex items-center justify-between text-sm">
                  <div>
                    <p className="font-medium text-[var(--text-primary)]">{n.name}</p>
                    <p className="font-mono text-xs text-[var(--text-muted)]">{n.cidr}</p>
                  </div>
                  <Badge tone={n.protected ? 'ok' : 'neutral'} dot>
                    {n.protected ? 'protected' : 'unmonitored'}
                  </Badge>
                </li>
              ))}
            </ul>
          </StatCard>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard title="Active containers" icon={Boxes} tone="brand">
            <ul className="space-y-2.5">
              {d.containers.map((c) => (
                <li key={c.name} className="flex items-center justify-between text-sm">
                  <span className="truncate text-[var(--text-secondary)]">{c.name}</span>
                  <Badge
                    tone={c.status === 'running' ? 'ok' : c.status === 'error' ? 'error' : 'neutral'}
                    dot
                  >
                    {c.status}
                  </Badge>
                </li>
              ))}
            </ul>
          </StatCard>

          <StatCard title="Training / lab-adapt" icon={GraduationCap} tone={d.training.active ? 'warn' : 'ok'}>
            {d.training.active ? (
              <div>
                <Badge tone="warn" dot>
                  running
                </Badge>
                <p className="mt-2 text-sm text-[var(--text-secondary)]">{d.training.job}</p>
              </div>
            ) : (
              <div>
                <Badge tone="ok" dot>
                  idle
                </Badge>
                <p className="mt-2 text-sm text-[var(--text-muted)]">{d.training.message}</p>
              </div>
            )}
          </StatCard>

          <StatCard title="Models" icon={BrainCircuit} tone="info">
            <ul className="space-y-2.5">
              {d.models.map((m) => (
                <li key={m.id} className="flex items-center justify-between text-sm">
                  <span className="text-[var(--text-secondary)]">
                    {m.name} <span className="text-[var(--text-muted)]">{m.version}</span>
                  </span>
                  <Badge tone={m.loaded ? 'ok' : 'neutral'} dot>
                    {m.loaded ? 'loaded' : 'idle'}
                  </Badge>
                </li>
              ))}
            </ul>
          </StatCard>

          <StatCard title="Datasets" icon={Database} tone="brand">
            <ul className="space-y-3">
              {d.datasets.map((ds) => (
                <li key={ds.name} className="text-sm">
                  <p className="font-medium text-[var(--text-primary)]">{ds.name}</p>
                  <p className="text-xs text-[var(--text-muted)]">
                    {ds.samples?.toLocaleString()} samples · {ds.sizeMb} MB
                  </p>
                </li>
              ))}
            </ul>
          </StatCard>
        </div>
      </div>
    </div>
  )
}
