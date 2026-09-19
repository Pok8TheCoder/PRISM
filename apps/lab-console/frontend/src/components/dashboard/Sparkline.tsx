import { useEffect, useRef } from 'react'
import type { MetricsPoint } from '../../types/dashboard'

interface SparklineProps {
  data: MetricsPoint[]
  field: 'flowRate' | 'pAttack'
  height?: number
  color?: string
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
}

export function Sparkline({
  data,
  field,
  height = 64,
  color,
}: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current

    if (!canvas || !container || data.length < 2) return

    const drawSparkline = () => {
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

      const values = data.map((point) => point[field])
      const min = Math.min(...values)
      const max = Math.max(...values)

      const padX = 2
      const padY = 5
      const plotWidth = width - padX * 2
      const plotHeight = height - padY * 2

      const lineColor = color ?? cssVar('--accent')

      const xAt = (index: number) =>
        padX +
        (index / (data.length - 1)) *
          plotWidth

      const yAt = (value: number) =>
        padY +
        plotHeight -
        ((value - min) /
          (max - min || 1)) *
          plotHeight

      const points = data.map((point, index) => ({
        x: xAt(index),
        y: yAt(point[field]),
      }))

      /* ------------------------------------------
         Gradient fill
      ------------------------------------------ */

      const gradient = ctx.createLinearGradient(
        0,
        padY,
        0,
        padY + plotHeight,
      )

      gradient.addColorStop(
        0,
        `${lineColor}28`,
      )

      gradient.addColorStop(
        1,
        `${lineColor}00`,
      )

      ctx.beginPath()

      ctx.moveTo(
        points[0].x,
        height,
      )

      points.forEach((point) => {
        ctx.lineTo(
          point.x,
          point.y,
        )
      })

      ctx.lineTo(
        points[points.length - 1].x,
        height,
      )

      ctx.closePath()

      ctx.fillStyle = gradient
      ctx.fill()

      /* ------------------------------------------
         Main line
      ------------------------------------------ */

      ctx.beginPath()

      points.forEach((point, index) => {
        if (index === 0) {
          ctx.moveTo(point.x, point.y)
        } else {
          ctx.lineTo(point.x, point.y)
        }
      })

      ctx.strokeStyle = lineColor
      ctx.lineWidth = 1.75
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'

      ctx.shadowColor = lineColor
      ctx.shadowBlur = 5

      ctx.stroke()

      ctx.shadowBlur = 0

      /* ------------------------------------------
         Latest value indicator
      ------------------------------------------ */

      const lastPoint = points[points.length - 1]

      ctx.beginPath()

      ctx.arc(
        lastPoint.x,
        lastPoint.y,
        3,
        0,
        Math.PI * 2,
      )

      ctx.fillStyle = lineColor
      ctx.fill()

      ctx.beginPath()

      ctx.arc(
        lastPoint.x,
        lastPoint.y,
        6,
        0,
        Math.PI * 2,
      )

      ctx.strokeStyle = `${lineColor}45`
      ctx.lineWidth = 1

      ctx.stroke()
    }

    drawSparkline()

    const resizeObserver = new ResizeObserver(() => {
      drawSparkline()
    })

    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
    }
  }, [data, field, height, color])

  return (
    <div
      ref={containerRef}
      className="w-full overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        className="block w-full"
      />
    </div>
  )
}