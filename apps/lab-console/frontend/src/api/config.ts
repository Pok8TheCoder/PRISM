/** API base URL. Defaults to `/api` (Vite dev proxy). Override in `.env`. */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

/** When true, hooks skip network calls and use in-app mocks only. */
export const FORCE_MOCK = import.meta.env.VITE_FORCE_MOCK === 'true'

export const DEV_PROXY_TARGET = import.meta.env.VITE_PROXY_TARGET ?? 'http://127.0.0.1:8790'
