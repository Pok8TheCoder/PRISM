import { useEffect, useRef } from 'react'
import type { MetricsPoint } from '../../types/dashboard'

interface AreaChartProps {
  data: MetricsPoint[]
  height?: number
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
}

export function AreaChart({
  data,
  height = 220,
}: AreaChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current

    if (!canvas || !container || data.length < 2) return

    const drawChart = () => {
      const dpr = window.devicePixelRatio || 1
      const width = container.clientWidth

      if (width <= 0) return

      canvas.width = width * dpr
      canvas.height = height * dpr
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)

      const pad = {
        l: 8,
        r: 8,
        t: 14,
        b: 24,
      }

      const plotW = width - pad.l - pad.r
      const plotH = height - pad.t - pad.b

      const accent = cssVar('--accent')
      const accent2 = cssVar('--accent-2')
      const grid = cssVar('--chart-grid')

      /*
       * Grid
       */
      ctx.strokeStyle = grid
      ctx.lineWidth = 1

      for (let i = 0; i <= 3; i++) {
        const y = pad.t + (plotH * i) / 3

        ctx.beginPath()
        ctx.moveTo(pad.l, y)
        ctx.lineTo(pad.l + plotW, y)
        ctx.stroke()
      }

      /*
       * Flow rate
       */
      const flowVals = data.map((point) => point.flowRate)
      const minFlow = Math.min(...flowVals)
      const maxFlow = Math.max(...flowVals)

      const xAt = (index: number) =>
        pad.l +
        (index / (data.length - 1)) * plotW

      const yFlow = (value: number) =>
        pad.t +
        plotH -
        ((value - minFlow) /
          (maxFlow - minFlow || 1)) *
          plotH

      /*
       * Flow area
       */
      const flowGradient = ctx.createLinearGradient(
        0,
        pad.t,
        0,
        pad.t + plotH,
      )

      flowGradient.addColorStop(
        0,
        `${accent}30`,
      )

      flowGradient.addColorStop(
        1,
        `${accent}00`,
      )

      ctx.beginPath()
      ctx.moveTo(
        xAt(0),
        pad.t + plotH,
      )

      data.forEach((point, index) => {
        ctx.lineTo(
          xAt(index),
          yFlow(point.flowRate),
        )
      })

      ctx.lineTo(
        xAt(data.length - 1),
        pad.t + plotH,
      )

      ctx.closePath()
      ctx.fillStyle = flowGradient
      ctx.fill()

      /*
       * Flow line
       */
      ctx.beginPath()

      data.forEach((point, index) => {
        const x = xAt(index)
        const y = yFlow(point.flowRate)

        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.strokeStyle = accent
      ctx.lineWidth = 2.25
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'

      ctx.shadowColor = accent
      ctx.shadowBlur = 6

      ctx.stroke()

      ctx.shadowBlur = 0

      /*
       * P(attack) overlay
       */
      ctx.beginPath()

      data.forEach((point, index) => {
        const x = xAt(index)

        const y =
          pad.t +
          plotH -
          point.pAttack * plotH

        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.strokeStyle = accent2
      ctx.lineWidth = 1.75
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.setLineDash([5, 4])

      ctx.stroke()

      ctx.setLineDash([])

      /*
       * Current flow point
       */
      const lastIndex = data.length - 1
      const lastPoint = data[lastIndex]

      const lastX = xAt(lastIndex)
      const lastY = yFlow(lastPoint.flowRate)

      ctx.beginPath()
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2)

      ctx.fillStyle = accent
      ctx.fill()

      ctx.beginPath()
      ctx.arc(lastX, lastY, 7, 0, Math.PI * 2)

      ctx.strokeStyle = `${accent}55`
      ctx.lineWidth = 1

      ctx.stroke()

      /*
       * Current P(attack) point
       */
      const attackY =
        pad.t +
        plotH -
        lastPoint.pAttack * plotH

      ctx.beginPath()
      ctx.arc(
        lastX,
        attackY,
        3.5,
        0,
        Math.PI * 2,
      )

      ctx.fillStyle = accent2
      ctx.fill()
    }

    drawChart()

    const resizeObserver = new ResizeObserver(() => {
      drawChart()
    })

    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
    }
  }, [data, height])

  return (
    <div
      ref={containerRef}
      className="w-full"
    >
      <canvas
        ref={canvasRef}
        className="block w-full"
      />

      <div className="mt-2 flex items-center gap-5 text-[11px] text-[var(--text-muted)]">
        <span className="inline-flex items-center gap-1.5">
          <span
            className="h-0.5 w-3 rounded-full"
            style={{
              background: 'var(--accent)',
            }}
          />
          Flow rate
        </span>

        <span className="inline-flex items-center gap-1.5">
          <span
            className="h-0.5 w-3 rounded-full"
            style={{
              background:
                'repeating-linear-gradient(90deg, var(--accent-2) 0 4px, transparent 4px 7px)',
            }}
          />
          P(attack)
        </span>
      </div>
    </div>
  )
}