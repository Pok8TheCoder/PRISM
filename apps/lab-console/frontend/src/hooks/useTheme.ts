import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'
export type AppearanceStyle = 'mono' | 'aurora'

const THEME_KEY = 'lab-console-theme'
const STYLE_KEY = 'lab-console-style'

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem(THEME_KEY) as Theme | null
    if (stored === 'light' || stored === 'dark') return stored
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  const [style, setStyleState] = useState<AppearanceStyle>(() => {
    const stored = localStorage.getItem(STYLE_KEY) as AppearanceStyle | null
    return stored === 'mono' || stored === 'aurora' ? stored : 'mono'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  useEffect(() => {
    document.documentElement.setAttribute('data-style', style)
    localStorage.setItem(STYLE_KEY, style)
  }, [style])

  const toggle = useCallback(() => {
    setThemeState((t) => (t === 'dark' ? 'light' : 'dark'))
  }, [])

  const setTheme = useCallback((t: Theme) => setThemeState(t), [])
  const setStyle = useCallback((s: AppearanceStyle) => setStyleState(s), [])
  const toggleStyle = useCallback(() => {
    setStyleState((s) => (s === 'mono' ? 'aurora' : 'mono'))
  }, [])

  return { theme, toggle, setTheme, style, setStyle, toggleStyle }
}
