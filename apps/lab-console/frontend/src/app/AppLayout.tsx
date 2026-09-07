import { Outlet } from 'react-router-dom'
import { useTheme } from '../hooks/useTheme'
import { Sidebar } from './Sidebar'

export function AppLayout() {
  const { theme, toggle, style, setStyle } = useTheme()

  return (
    <div className="flex h-full">
      <Sidebar theme={theme} onToggleTheme={toggle} style={style} onSetStyle={setStyle} />
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <Outlet />
      </div>
    </div>
  )
}
