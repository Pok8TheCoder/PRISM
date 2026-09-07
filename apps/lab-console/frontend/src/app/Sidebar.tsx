import { Link, useLocation } from 'react-router-dom'
import clsx from 'clsx'
import {
  LayoutDashboard,
  PlaySquare,
  FlaskConical,
  FileSearch,
  Video,
  Moon,
  Sun,
  ShieldHalf,
  Contrast,
  Sparkles,
} from 'lucide-react'
import type { AppearanceStyle, Theme } from '../hooks/useTheme'

const nav = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/session', label: 'Lab Session', icon: PlaySquare },
  { to: '/adversarial', label: 'Adversarial Lab', icon: FlaskConical },
  { to: '/explain', label: 'Explain', icon: FileSearch, placeholder: true },
  { to: '/recordings', label: 'Recordings', icon: Video, placeholder: true },
]

interface SidebarProps {
  theme: Theme
  onToggleTheme: () => void
  style: AppearanceStyle
  onSetStyle: (style: AppearanceStyle) => void
}

export function Sidebar({ theme, onToggleTheme, style, onSetStyle }: SidebarProps) {
  const location = useLocation()

  return (
    <aside className="flex w-[224px] shrink-0 flex-col border-r border-[var(--border)] bg-[var(--bg-surface)]/60 px-3 py-4 backdrop-blur-xl">
      <div className="mb-6 flex items-center gap-2.5 px-2">
        <div
          className="flex h-9 w-9 items-center justify-center rounded-xl"
          style={{ background: 'var(--gradient-brand)', boxShadow: '0 4px 18px -4px var(--accent-glow)' }}
        >
          <ShieldHalf size={19} color="white" strokeWidth={2.3} />
        </div>
        <div className="min-w-0">
          <p className="truncate text-[15px] font-bold leading-tight tracking-tight">PRISM</p>
          <p className="truncate text-[11px] leading-tight text-[var(--text-muted)]">Lab Console</p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {nav.map((item) => {
          const active = location.pathname === item.to
          const Icon = item.icon
          return (
            <Link
              key={item.to}
              to={item.to}
              className={clsx(
                'group relative flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium transition-all',
                active
                  ? 'text-[var(--text-primary)]'
                  : 'text-[var(--text-muted)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]',
                item.placeholder && 'opacity-55',
              )}
              style={
                active
                  ? { background: 'var(--accent-soft)', boxShadow: 'inset 0 0 0 1px var(--border-strong)' }
                  : undefined
              }
            >
              {active && (
                <span
                  className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full"
                  style={{ background: 'var(--gradient-brand)' }}
                />
              )}
              <Icon size={17} strokeWidth={2.1} className={active ? 'text-[var(--accent)]' : ''} />
              <span className="truncate">{item.label}</span>
              {item.placeholder && (
                <span className="ml-auto rounded-full bg-[var(--bg-elevated)] px-1.5 py-0.5 text-[9px] font-semibold uppercase text-[var(--text-muted)]">
                  soon
                </span>
              )}
            </Link>
          )
        })}
      </nav>

      <div className="flex flex-col gap-2">
        <div className="flex gap-1 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-1">
          <button
            type="button"
            onClick={() => onSetStyle('mono')}
            title="Mono — formal black & white"
            className={clsx(
              'flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-colors',
              style === 'mono'
                ? 'text-[var(--bg-page)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]',
            )}
            style={style === 'mono' ? { background: 'var(--gradient-brand)' } : undefined}
          >
            <Contrast size={13} /> Mono
          </button>
          <button
            type="button"
            onClick={() => onSetStyle('aurora')}
            title="Aurora — violet/cyan gradient"
            className={clsx(
              'flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-colors',
              style === 'aurora' ? 'text-white' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]',
            )}
            style={
              style === 'aurora'
                ? { background: 'linear-gradient(135deg, #8b5cf6, #22d3ee)' }
                : undefined
            }
          >
            <Sparkles size={13} /> Aurora
          </button>
        </div>

        <button
          type="button"
          onClick={onToggleTheme}
          className="flex items-center gap-2.5 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-2 text-sm font-medium text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)]"
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
      </div>
    </aside>
  )
}
