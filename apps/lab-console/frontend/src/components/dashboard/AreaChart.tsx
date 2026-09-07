import { useEffect, useRef } from 'react'
import type { MetricsPoint } from '../../types/dashboard'

interface AreaChartProps {
  data: MetricsPoint[]
  height?: number
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function AreaChart({ data, height = 220 }: AreaChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || data.length < 2) return

    const dpr = window.devicePixelRatio || 1
    const width = container.clientWidth
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)

    const pad = { l: 8, r: 8, t: 16, b: 24 }
    const plotW = width - pad.l - pad.r
    const plotH = height - pad.t - pad.b

    // Grid lines
    ctx.strokeStyle = cssVar('--chart-grid')
    ctx.lineWidth = 1
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (plotH * i) / 3
      ctx.beginPath()
      ctx.moveTo(pad.l, y)
      ctx.lineTo(pad.l + plotW, y)
      ctx.stroke()
    }

    const flowVals = data.map((d) => d.flowRate)
    const minF = Math.min(...flowVals)
    const maxF = Math.max(...flowVals)
    const xAt = (i: number) => pad.l + (i / (data.length - 1)) * plotW
    const yFlow = (v: number) => pad.t + plotH - ((v - minF) / (maxF - minF || 1)) * plotH

    // Flow rate area — brand gradient fill
    const accent = cssVar('--accent')
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH)
    grad.addColorStop(0, accent + '40')
    grad.addColorStop(1, accent + '00')

    ctx.beginPath()
    ctx.moveTo(xAt(0), pad.t + plotH)
    data.forEach((d, i) => ctx.lineTo(xAt(i), yFlow(d.flowRate)))
    ctx.lineTo(xAt(data.length - 1), pad.t + plotH)
    ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()

    ctx.beginPath()
    data.forEach((d, i) => {
      const x = xAt(i)
      const y = yFlow(d.flowRate)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.strokeStyle = accent
    ctx.lineWidth = 2.5
    ctx.lineJoin = 'round'
    ctx.shadowColor = accent
    ctx.shadowBlur = 10
    ctx.stroke()
    ctx.shadowBlur = 0

    // P(attack) overlay line — cyan
    const cyan = cssVar('--accent-2')
    ctx.beginPath()
    data.forEach((d, i) => {
      const x = xAt(i)
      const y = pad.t + plotH - d.pAttack * plotH
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.strokeStyle = cyan
    ctx.lineWidth = 2
    ctx.setLineDash([5, 4])
    ctx.stroke()
    ctx.setLineDash([])
  }, [data, height])

  return (
    <div ref={containerRef} className="w-full">
      <canvas ref={canvasRef} className="block w-full" />
      <div className="mt-1 flex gap-4 text-[11px] text-[var(--text-muted)]">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-3 rounded-full" style={{ background: 'var(--accent)' }} /> Flow rate
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="h-0.5 w-3 rounded-full"
            style={{ background: 'repeating-linear-gradient(90deg, var(--accent-2) 0 4px, transparent 4px 7px)' }}
          />
          P(attack)
        </span>
      </div>
    </div>
  )
}
