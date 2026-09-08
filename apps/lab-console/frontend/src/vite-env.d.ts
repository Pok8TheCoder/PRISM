/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** REST prefix, e.g. `/api` or `http://127.0.0.1:8790/api` */
  readonly VITE_API_BASE_URL?: string
  /** Vite dev-server proxy target for `/api` (default `http://127.0.0.1:8790`) */
  readonly VITE_PROXY_TARGET?: string
  /** Set `true` to always use in-app mocks (no BFF required) */
  readonly VITE_FORCE_MOCK?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
