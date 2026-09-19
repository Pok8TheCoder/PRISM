import { Outlet } from 'react-router-dom'
import { useTheme } from '../hooks/useTheme'
import { Sidebar } from './Sidebar'

export function AppLayout() {
  const { theme, toggle, style, setStyle } = useTheme()

  return (
    <div className="flex h-full overflow-hidden bg-[var(--bg-page)]">
      <Sidebar
        theme={theme}
        onToggleTheme={toggle}
        style={style}
        onSetStyle={setStyle}
      />

      <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-[var(--bg-page)]">
        <Outlet />
      </main>
    </div>
  )
}
