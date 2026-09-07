import { useEffect, useRef } from 'react'
import type { MetricsPoint } from '../../types/dashboard'

interface SparklineProps {
  data: MetricsPoint[]
  field: 'flowRate' | 'pAttack'
  height?: number
  color?: string
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function Sparkline({ data, field, height = 64, color }: SparklineProps) {
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

    const vals = data.map((d) => d[field])
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const pad = 4
    const lineColor = color ?? cssVar('--accent')

    const pts = data.map((d, i) => ({
      x: (i / (data.length - 1)) * width,
      y: height - pad - ((d[field] - min) / (max - min || 1)) * (height - pad * 2),
    }))

    // Gradient fill under curve
    const grad = ctx.createLinearGradient(0, 0, 0, height)
    grad.addColorStop(0, lineColor + '33')
    grad.addColorStop(1, lineColor + '00')

    ctx.beginPath()
    ctx.moveTo(pts[0].x, height)
    pts.forEach((p) => ctx.lineTo(p.x, p.y))
    ctx.lineTo(pts[pts.length - 1].x, height)
    ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()

    // Line with glow
    ctx.beginPath()
    pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)))
    ctx.strokeStyle = lineColor
    ctx.lineWidth = 2
    ctx.lineJoin = 'round'
    ctx.shadowColor = lineColor
    ctx.shadowBlur = 8
    ctx.stroke()
    ctx.shadowBlur = 0

    // End dot
    const last = pts[pts.length - 1]
    ctx.beginPath()
    ctx.arc(last.x, last.y, 3, 0, Math.PI * 2)
    ctx.fillStyle = lineColor
    ctx.shadowColor = lineColor
    ctx.shadowBlur = 10
    ctx.fill()
    ctx.shadowBlur = 0
  }, [data, field, height, color])

  return (
    <div ref={containerRef} className="w-full">
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  )
}
