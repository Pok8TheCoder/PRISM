import { useCallback, useEffect, useRef, useState } from 'react'
import {
  checkApiHealth,
  fetchDashboard,
  fetchLabConfig,
  fetchLogs,
  fetchModels,
  fetchScripts,
  fetchSession,
  patchLabConfig,
  patchSessionPolicy,
  runScript,
  FORCE_MOCK,
} from '../api'
import { mockDashboard } from '../mocks/dashboard'
import { mockModelRegistry } from '../mocks/models'
import { mockLogs, mockScripts, mockSession } from '../mocks/session'
import type { DashboardState } from '../types/dashboard'
import type { ModelRegistryEntry } from '../types/models'
import type { LogLine, PolicyMode, ScriptDef, SessionMode, SessionState } from '../types/session'

export function useApiAvailable() {
  const [available, setAvailable] = useState<boolean | null>(null)

  useEffect(() => {
    checkApiHealth().then(setAvailable)
  }, [])

  return available
}

export function useDashboardData() {
  const [data, setData] = useState<DashboardState>(mockDashboard)
  const [loading, setLoading] = useState(true)
  const [source, setSource] = useState<'api' | 'mock'>('mock')

  const refresh = useCallback(async () => {
    if (FORCE_MOCK) {
      setData(mockDashboard)
      setSource('mock')
      setLoading(false)
      return
    }
    try {
      const d = await fetchDashboard()
      setData(d)
      setSource('api')
    } catch {
      setData(mockDashboard)
      setSource('mock')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = window.setInterval(refresh, 5000)
    return () => window.clearInterval(id)
  }, [refresh])

  return { data, loading, source, refresh }
}

export function useSessionData(sessionId: string, mode: SessionMode) {
  const [session, setSession] = useState<SessionState>(mockSession)
  const [loading, setLoading] = useState(true)
  const [source, setSource] = useState<'api' | 'mock'>('mock')

  const refresh = useCallback(async () => {
    if (FORCE_MOCK) {
      setSession({ ...mockSession, mode })
      setSource('mock')
      setLoading(false)
      return
    }
    try {
      const s = await fetchSession(sessionId, mode)
      setSession(s)
      setSource('api')
    } catch {
      setSession({ ...mockSession, mode })
      setSource('mock')
    } finally {
      setLoading(false)
    }
  }, [sessionId, mode])

  useEffect(() => {
    refresh()
    const ms = mode === 'live' ? 250 : 5000
    const id = window.setInterval(refresh, ms)
    return () => window.clearInterval(id)
  }, [refresh, mode])

  const setPolicy = useCallback(
    async (policyMode: PolicyMode) => {
      setSession((s) => ({ ...s, policyMode }))
      try {
        await patchSessionPolicy(sessionId, policyMode)
      } catch {
        /* local-only when API offline */
      }
    },
    [sessionId],
  )

  return { session, setSession, loading, source, refresh, setPolicy }
}

export function useLogStream() {
  const [logs, setLogs] = useState<LogLine[]>(() => (FORCE_MOCK ? mockLogs : []))
  const sinceRef = useRef(0)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const lines = await fetchLogs(sinceRef.current)
        if (cancelled || lines.length === 0) return
        const maxTs = Math.max(...lines.map((l) => l.ts))
        sinceRef.current = Math.max(sinceRef.current, maxTs)
        setLogs((prev) => {
          const seen = new Set(prev.map((p) => `${p.ts}:${p.text}`))
          const merged = [...prev]
          for (const ln of lines) {
            const key = `${ln.ts}:${ln.text}`
            if (!seen.has(key)) merged.push(ln)
          }
          return merged.slice(-500)
        })
      } catch {
        /* keep mock logs */
      }
    }
    poll()
    const id = window.setInterval(poll, 1500)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  return logs
}

export function useScripts() {
  const [scripts, setScripts] = useState<ScriptDef[]>(mockScripts)

  useEffect(() => {
    fetchScripts()
      .then(setScripts)
      .catch(() => setScripts(mockScripts))
  }, [])

  const launch = useCallback(async (id: string, delaySec?: number) => {
    try {
      await runScript(id, { delaySec })
    } catch {
      /* ignore */
    }
  }, [])

  return { scripts, launch }
}

export function useLabConfig() {
  const [config, setConfig] = useState({ horizonSec: 60, attackDelaySec: null as number | null })

  useEffect(() => {
    fetchLabConfig()
      .then((c) => setConfig({ horizonSec: c.horizonSec ?? 60, attackDelaySec: c.attackDelaySec ?? null }))
      .catch(() => {})
  }, [])

  const update = useCallback(async (patch: { horizonSec?: number; attackDelaySec?: number | null }) => {
    setConfig((prev) => ({ ...prev, ...patch }))
    try {
      const c = await patchLabConfig(patch)
      setConfig({ horizonSec: c.horizonSec ?? 60, attackDelaySec: c.attackDelaySec ?? null })
    } catch {
      /* local-only */
    }
  }, [])

  return { config, update }
}

export function useModelRegistry() {
  const [models, setModels] = useState<ModelRegistryEntry[]>(mockModelRegistry)
  const [loading, setLoading] = useState(true)
  const [source, setSource] = useState<'api' | 'mock'>('mock')

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m)
        setSource('api')
      })
      .catch(() => {
        setModels(mockModelRegistry)
        setSource('mock')
      })
      .finally(() => setLoading(false))
  }, [])

  return { models, loading, source }
}
