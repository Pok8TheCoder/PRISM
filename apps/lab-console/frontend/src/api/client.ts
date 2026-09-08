import type { DashboardState } from '../types/dashboard'
import type { ModelRegistryEntry } from '../types/models'
import type { LogLine, ScriptDef, SessionState } from '../types/session'
import { API_BASE } from './config'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export async function fetchDashboard(): Promise<DashboardState> {
  return request<DashboardState>('/dashboard')
}

export async function fetchSession(sessionId: string, mode: 'live' | 'recorded'): Promise<SessionState> {
  return request<SessionState>(`/sessions/${sessionId}/state?mode=${mode}`)
}

export async function patchSessionPolicy(sessionId: string, policyMode: 'ids' | 'ips'): Promise<{ policyMode: string }> {
  return request(`/sessions/${sessionId}/policy`, {
    method: 'PATCH',
    body: JSON.stringify({ policyMode }),
  })
}

export async function fetchLogs(since = 0): Promise<LogLine[]> {
  const data = await request<{ lines: LogLine[] }>(`/logs?since=${since}`)
  return data.lines
}

export async function fetchScripts(): Promise<ScriptDef[]> {
  const data = await request<{ scripts: ScriptDef[] }>('/scripts')
  return data.scripts
}

export async function runScript(
  scriptId: string,
  opts?: { delaySec?: number },
): Promise<{ job_id: string; status: string }> {
  return request('/scripts/run', {
    method: 'POST',
    body: JSON.stringify({ script_id: scriptId, delay_sec: opts?.delaySec }),
  })
}

export interface LabConfig {
  horizonSec: number
  attackDelaySec: number | null
  attackDelayMinSec?: number
  attackDelayMaxSec?: number
}

export async function fetchLabConfig(): Promise<LabConfig> {
  return request<LabConfig>('/lab-config')
}

export async function patchLabConfig(patch: Partial<LabConfig>): Promise<LabConfig> {
  return request<LabConfig>('/lab-config', {
    method: 'PATCH',
    body: JSON.stringify(patch),
  })
}

export async function fetchModels(): Promise<ModelRegistryEntry[]> {
  const data = await request<{ models: ModelRegistryEntry[] }>('/models')
  return data.models
}

export interface ApiHealth {
  ok: boolean
  apiVersion?: string
  features?: { terminalPty?: boolean; liveCharts?: boolean }
}

export async function fetchApiHealth(): Promise<ApiHealth> {
  return request<ApiHealth>('/health')
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const h = await fetchApiHealth()
    return Boolean(h.ok)
  } catch {
    return false
  }
}

/** WebSocket URL for interactive shell (proxied via Vite in dev). */
export function terminalWsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/api/terminal`
}
