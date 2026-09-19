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

  const runningCount = d.containers.filter(
    (c) => c.status === 'running',
  ).length

  const loadedModels = d.models.filter(
    (m) => m.loaded,
  ).length

  const latestP =
    d.metricsTimeseries[
      d.metricsTimeseries.length - 1
    ]?.pAttack ?? 0

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <TopBar
        title="Dashboard"
        subtitle="Live network activity & background job insights"
        right={<ApiSourceBadge source={source} />}
      />

      <div className="flex-1 overflow-y-auto px-5 py-5 lg:px-6 lg:py-6">
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
            delta={
              latestP > 0.5
                ? 'elevated'
                : 'nominal'
            }
            deltaTone={
              latestP > 0.5
                ? 'error'
                : 'ok'
            }
            icon={ShieldAlert}
            tone={
              latestP > 0.5
                ? 'error'
                : 'ok'
            }
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
            value={
              d.device.cuda
                ? `${d.device.vramGb} GB`
                : 'CPU only'
            }
            delta={d.device.deviceName}
            deltaTone="ok"
            icon={Cpu}
            tone="brand"
          />
        </div>

        {/* Main monitoring section */}
        <div className="mb-6 grid gap-4 xl:grid-cols-3">
          {/* Primary monitoring panel */}
          <StatCard
            title="Network activity"
            icon={Activity}
            tone="brand"
            className="xl:col-span-2"
          >
            <div className="pt-1">
              <AreaChart
                data={d.metricsTimeseries}
                height={240}
              />
            </div>
          </StatCard>

          {/* Protected networks */}
          <StatCard
            title="Protected networks"
            icon={Network}
            tone="info"
          >
            <ul className="divide-y divide-[var(--border)]">
              {d.networks.map((n) => (
                <li
                  key={n.name}
                  className="flex items-center justify-between gap-3 py-3 first:pt-1 last:pb-1"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-[var(--text-primary)]">
                      {n.name}
                    </p>

                    <p className="mt-0.5 font-mono text-[11px] text-[var(--text-muted)]">
                      {n.cidr}
                    </p>
                  </div>

                  <Badge
                    tone={
                      n.protected
                        ? 'ok'
                        : 'neutral'
                    }
                    dot
                  >
                    {n.protected
                      ? 'protected'
                      : 'unmonitored'}
                  </Badge>
                </li>
              ))}
            </ul>
          </StatCard>
        </div>

        {/* Secondary system overview */}
        <div className="mb-3 flex items-center justify-between px-1">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
              System overview
            </p>

            <p className="mt-1 text-xs text-[var(--text-muted)]">
              Runtime, training, model and dataset status
            </p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {/* Active containers */}
          <StatCard
            title="Active containers"
            icon={Boxes}
            tone="brand"
            className="rounded-lg bg-[var(--bg-surface)]/50 shadow-none hover:-translate-y-0 hover:shadow-none"
          >
            <ul className="divide-y divide-[var(--border)]">
              {d.containers.map((c) => (
                <li
                  key={c.name}
                  className="flex items-center justify-between gap-3 py-2.5 first:pt-1 last:pb-1"
                >
                  <span className="truncate text-sm text-[var(--text-secondary)]">
                    {c.name}
                  </span>

                  <Badge
                    tone={
                      c.status === 'running'
                        ? 'ok'
                        : c.status === 'error'
                          ? 'error'
                          : 'neutral'
                    }
                    dot
                  >
                    {c.status}
                  </Badge>
                </li>
              ))}
            </ul>
          </StatCard>

          {/* Training */}
          <StatCard
            title="Training / lab-adapt"
            icon={GraduationCap}
            tone={
              d.training.active
                ? 'warn'
                : 'ok'
            }
            className="rounded-lg bg-[var(--bg-surface)]/50 shadow-none hover:-translate-y-0 hover:shadow-none"
          >
            {d.training.active ? (
              <div className="space-y-3">
                <Badge tone="warn" dot>
                  running
                </Badge>

                <p className="text-sm leading-5 text-[var(--text-secondary)]">
                  {d.training.job}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <Badge tone="ok" dot>
                  idle
                </Badge>

                <p className="text-sm leading-5 text-[var(--text-muted)]">
                  {d.training.message}
                </p>
              </div>
            )}
          </StatCard>

          {/* Models */}
          <StatCard
            title="Models"
            icon={BrainCircuit}
            tone="info"
            className="rounded-lg bg-[var(--bg-surface)]/50 shadow-none hover:-translate-y-0 hover:shadow-none"
          >
            <ul className="divide-y divide-[var(--border)]">
              {d.models.map((m) => (
                <li
                  key={m.id}
                  className="flex items-center justify-between gap-3 py-2.5 first:pt-1 last:pb-1"
                >
                  <span className="truncate text-sm text-[var(--text-secondary)]">
                    {m.name}{' '}
                    <span className="text-[var(--text-muted)]">
                      {m.version}
                    </span>
                  </span>

                  <Badge
                    tone={
                      m.loaded
                        ? 'ok'
                        : 'neutral'
                    }
                    dot
                  >
                    {m.loaded
                      ? 'loaded'
                      : 'idle'}
                  </Badge>
                </li>
              ))}
            </ul>
          </StatCard>

          {/* Datasets */}
          <StatCard
            title="Datasets"
            icon={Database}
            tone="brand"
            className="rounded-lg bg-[var(--bg-surface)]/50 shadow-none hover:-translate-y-0 hover:shadow-none"
          >
            <ul className="divide-y divide-[var(--border)]">
              {d.datasets.map((ds) => (
                <li
                  key={ds.name}
                  className="py-2.5 first:pt-1 last:pb-1"
                >
                  <p className="truncate text-sm font-medium text-[var(--text-primary)]">
                    {ds.name}
                  </p>

                  <p className="mt-1 text-[11px] text-[var(--text-muted)]">
                    {ds.samples?.toLocaleString()}{' '}
                    samples · {ds.sizeMb} MB
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