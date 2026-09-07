import { useEffect, useRef } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'

interface MockTerminalProps {
  onCommand?: (cmd: string) => void
}

const MOCK_RESPONSES: Record<string, string> = {
  help: 'Commands: help, status, docker ps, killchain recon|enum|spray|loot',
  status: 'Lab: running · Scorer: idle · Policy: IDS',
  'docker ps': 'harborline-site  attacker-bot  suricata',
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/** Resolve any CSS color (hex, rgb, var()) to an [r, g, b] triple via computed style. */
function resolveRgb(cssColor: string): [number, number, number] {
  const el = document.createElement('div')
  el.style.color = cssColor
  document.body.appendChild(el)
  const rgb = getComputedStyle(el).color
  document.body.removeChild(el)
  const m = rgb.match(/\d+/g)
  if (!m) return [255, 255, 255]
  return [Number(m[0]), Number(m[1]), Number(m[2])]
}

export function MockTerminal({ onCommand }: MockTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const bg = cssVar('--terminal-bg') || '#05070d'
    const fg = cssVar('--terminal-fg') || '#cdd6f4'
    const accent = cssVar('--accent') || '#e8e8e8'
    const [ar, ag, ab] = resolveRgb('var(--accent)')
    const [a2r, a2g, a2b] = resolveRgb('var(--accent-2)')

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
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(el)
    fit.fit()

    term.writeln(`\x1b[38;2;${ar};${ag};${ab}mPRISM Lab Console\x1b[0m — mock terminal (phase 2: WS /api/terminal)`)
    term.writeln(`Type \x1b[38;2;${a2r};${a2g};${a2b}mhelp\x1b[0m for commands.\r\n`)
    let buffer = ''

    term.onData((data) => {
      if (data === '\r') {
        term.write('\r\n')
        const cmd = buffer.trim().toLowerCase()
        onCommand?.(cmd)
        const resp = MOCK_RESPONSES[cmd] ?? `mock: ${buffer.trim() || '(empty)'}`
        term.writeln(resp)
        buffer = ''
      } else if (data === '\u007f') {
        if (buffer.length > 0) {
          buffer = buffer.slice(0, -1)
          term.write('\b \b')
        }
      } else {
        buffer += data
        term.write(data)
      }
    })

    termRef.current = term
    const ro = new ResizeObserver(() => fit.fit())
    ro.observe(el)

    return () => {
      ro.disconnect()
      term.dispose()
    }
  }, [onCommand])

  return (
    <div
      ref={containerRef}
      className="h-40 min-h-[120px] rounded-xl border border-[var(--border)] p-2"
      style={{ background: 'var(--terminal-bg)' }}
    />
  )
}
