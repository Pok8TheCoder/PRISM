import { useEffect, useRef, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'
import { fetchApiHealth, terminalWsUrl } from '../../api/client'

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

type ConnState = 'connecting' | 'open' | 'closed' | 'error' | 'stale-api'

interface LabTerminalProps {
  autoScroll: boolean
}

export function LabTerminal({ autoScroll }: LabTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const autoScrollRef = useRef(autoScroll)
  const [state, setState] = useState<ConnState>('connecting')
  const [session, setSession] = useState(0)

  useEffect(() => {
    autoScrollRef.current = autoScroll
    if (autoScroll && termRef.current) {
      termRef.current.scrollToBottom()
    }
  }, [autoScroll])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    let cancelled = false
    setState('connecting')

    const start = async () => {
      try {
        const health = await fetchApiHealth()
        if (cancelled) return
        if (!health.features?.terminalPty) {
          setState('stale-api')
          return
        }
      } catch {
        if (!cancelled) setState('error')
        return
      }

      if (cancelled) return

      const bg = cssVar('--terminal-bg') || '#05070d'
      const fg = cssVar('--terminal-fg') || '#cdd6f4'
      const accent = cssVar('--accent') || '#e8e8e8'

      const term = new Terminal({
        theme: {
          background: bg,
          foreground: fg,
          cursor: accent,
          cursorAccent: bg,
          selectionBackground: cssVar('--accent-glow') || 'rgba(255,255,255,0.16)',
        },
        fontSize: 12,
        fontFamily: "'JetBrains Mono', Consolas, monospace",
        cursorBlink: true,
        lineHeight: 1.35,
        convertEol: true,
      })
      const fit = new FitAddon()
      term.loadAddon(fit)
      term.open(el)
      termRef.current = term
      fit.fit()

      const ws = new WebSocket(terminalWsUrl())
      ws.binaryType = 'arraybuffer'

      const sendResize = () => {
        if (ws.readyState !== WebSocket.OPEN) return
        ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }))
      }

      ws.onopen = () => {
        setState('open')
        fit.fit()
        sendResize()
        term.focus()
      }

      ws.onmessage = (ev) => {
        if (typeof ev.data === 'string') term.write(ev.data)
        else term.write(new Uint8Array(ev.data as ArrayBuffer))
        if (autoScrollRef.current) term.scrollToBottom()
      }

      ws.onerror = () => setState('error')
      ws.onclose = () => setState('closed')

      term.onData((data) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(data)
      })

      const ro = new ResizeObserver(() => {
        fit.fit()
        sendResize()
      })
      ro.observe(el)

      return () => {
        ro.disconnect()
        ws.close()
        term.dispose()
        termRef.current = null
      }
    }

    let cleanup: (() => void) | undefined
    start().then((fn) => {
      cleanup = fn
    })

    return () => {
      cancelled = true
      cleanup?.()
    }
  }, [session])

  const statusText = {
    open: 'connected · PRISM repo shell',
    connecting: 'connecting…',
    closed: 'disconnected',
    error: 'connection error — is the API running?',
    'stale-api': 'stale API on :8790 — restart: python run_lab_console.py --api',
  }[state]

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1">
      <div className="flex items-center justify-between px-0.5">
        <span
          className={`text-[10px] font-medium uppercase tracking-wide ${
            state === 'stale-api' ? 'text-[var(--badge-warn)]' : 'text-[var(--text-muted)]'
          }`}
        >
          {statusText}
        </span>
        {(state === 'closed' || state === 'error' || state === 'stale-api') && (
          <button
            type="button"
            onClick={() => setSession((n) => n + 1)}
            className="text-[10px] font-semibold text-[var(--accent)] hover:underline"
          >
            Reconnect
          </button>
        )}
      </div>
      <div
        ref={containerRef}
        className="min-h-[120px] flex-1 rounded-xl border border-[var(--border)] p-2"
        style={{ background: 'var(--terminal-bg)' }}
      />
    </div>
  )
}
