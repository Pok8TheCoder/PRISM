import { useEffect, useRef } from 'react'
import type { ModelAccuracy, ModelRegion, SeriesPoint } from '../../types/session'

export interface TimeSeriesChartProps {
  actual: SeriesPoint[]
  /** Per-model retrospective scored line (left of playhead). */
  observed?: SeriesPoint[]
  predicted: SeriesPoint[]
  modelRegions: ModelRegion[]
  groundTruthRegions: ModelRegion[]
  memoryRegions?: ModelRegion[]
  playheadSec: number
  /** Visible time span (seconds), centered on playheadSec. The "now" line stays fixed
   *  in the middle of the chart and the data scrolls underneath it, like a live monitor. */
  windowSec?: number
  /** Scorer window length — used to step future actual padding in live mode. */
  dataStepSec?: number
  accuracy?: ModelAccuracy
  height?: number
}

/** Pick a "nice" tick step so we get roughly 2 gridlines on each side of "now". */
function niceTickStep(halfWindow: number): number {
  const target = halfWindow / 2
  const candidates = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300]
  let best = candidates[0]
  for (const c of candidates) {
    if (c <= target) best = c
  }
  return best
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

function regionFill(region: ModelRegion, playhead: number, isOverlap: boolean): { fill: string; border: string } {
  if (isOverlap) return { fill: cssVar('--region-overlap'), border: cssVar('--region-overlap-border') }
  if (region.kind === 'episodic') return { fill: cssVar('--region-episodic'), border: cssVar('--region-episodic-border') }
  if (region.kind === 'ground_truth') return { fill: cssVar('--region-gt'), border: cssVar('--region-gt-border') }
  const past = region.end <= playhead
  if (region.kind === 'suspicious') {
    if (past && region.resolved && region.correct === false) {
      return { fill: cssVar('--region-suspicious-resolved'), border: cssVar('--region-suspicious-resolved-border') }
    }
    return { fill: cssVar('--region-suspicious'), border: cssVar('--region-suspicious-border') }
  }
  if (region.kind === 'attack') {
    if (past && region.resolved && region.correct === false) {
      return { fill: cssVar('--region-attack-resolved'), border: cssVar('--region-attack-resolved-border') }
    }
    return { fill: cssVar('--region-attack'), border: cssVar('--region-attack-border') }
  }
  return { fill: 'transparent', border: 'transparent' }
}

function overlaps(a: ModelRegion, b: ModelRegion): boolean {
  return a.start < b.end && b.start < a.end
}

/** Observed past + flat continuation through the visible window (attacks shown as region bands). */
function buildActualLine(
  actual: SeriesPoint[],
  playheadSec: number,
  minT: number,
  maxT: number,
  stepSec: number,
): SeriesPoint[] {
  const step = Math.max(stepSec * 0.5, 0.25)
  const past = actual.filter((p) => p.t <= playheadSec + 0.02).sort((a, b) => a.t - b.t)

  const out: SeriesPoint[] = []
  for (const p of past) {
    if (out.length && Math.abs(out[out.length - 1].t - p.t) < step * 0.2) continue
    out.push(p)
  }

  const baseline = out.at(-1)?.y ?? 0.06

  const lastT = out.at(-1)?.t
  if (lastT !== undefined && lastT < playheadSec - 0.01) {
    out.push({ t: playheadSec, y: baseline })
  }

  let t = Math.max(out.at(-1)?.t ?? playheadSec, playheadSec)
  while (t < maxT - 0.01) {
    t = Math.round((t + step) * 1000) / 1000
    if (out.some((p) => Math.abs(p.t - t) < step * 0.35)) continue
    out.push({ t, y: baseline })
  }

  return out.filter((p) => p.t >= minT - step && p.t <= maxT + step)
}

function modelPastPoints(observed: SeriesPoint[], predicted: SeriesPoint[], playheadSec: number): SeriesPoint[] {
  if (observed.length > 0) {
    return observed.filter((p) => p.t <= playheadSec + 0.05).sort((a, b) => a.t - b.t)
  }
  return predicted.filter((p) => p.t <= playheadSec + 0.05).sort((a, b) => a.t - b.t)
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, Math.abs(w) / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

export function TimeSeriesChart({
  actual,
  observed = [],
  predicted,
  modelRegions,
  groundTruthRegions,
  memoryRegions = [],
  playheadSec,
  windowSec = 24,
  dataStepSec = 1,
  accuracy,
  height = 220,
}: TimeSeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const draw = () => {
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

      const pad = { l: 40, r: 14, t: 30, b: 26 }
      const plotW = width - pad.l - pad.r
      const plotH = height - pad.t - pad.b

      // "Now" is always dead-center: the window scrolls underneath the fixed playhead line.
      const halfWindow = windowSec / 2
      const minT = playheadSec - halfWindow
      const maxT = playheadSec + halfWindow

      const pastModel = modelPastPoints(observed, predicted, playheadSec).filter(
        (p) => p.t >= minT - 0.5,
      )
      const futureForecast = predicted
        .filter((p) => p.t > playheadSec + 0.001 && p.t <= maxT + 0.5)
        .sort((a, b) => a.t - b.t)

      const fullActual = buildActualLine(actual, playheadSec, minT, maxT, dataStepSec)
      const pastActual = fullActual.filter((p) => p.t <= playheadSec + 0.02)

      // Scale from model lines + measured actual (not synthetic GT future spikes).
      const scaleYs = [
        ...pastModel.map((p) => p.y),
        ...futureForecast.map((p) => p.y),
        ...pastActual.map((p) => p.y),
      ]
      const minY = scaleYs.length ? Math.min(...scaleYs, 0) - 0.05 : -0.05
      const maxY = scaleYs.length ? Math.max(...scaleYs, 1) + 0.05 : 1.05

      const xScale = (t: number) => pad.l + ((t - minT) / (maxT - minT || 1)) * plotW
      const yScale = (y: number) => pad.t + plotH - ((y - minY) / (maxY - minY || 1)) * plotH

      // Background grid
      ctx.strokeStyle = cssVar('--chart-grid')
      ctx.lineWidth = 1
      ctx.font = '10px Inter, sans-serif'
      ctx.fillStyle = cssVar('--text-muted')
      for (let i = 0; i <= 3; i++) {
        const y = pad.t + (plotH * i) / 3
        ctx.beginPath()
        ctx.moveTo(pad.l, y)
        ctx.lineTo(pad.l + plotW, y)
        ctx.stroke()
        const val = maxY - ((maxY - minY) * i) / 3
        ctx.fillText(val.toFixed(2), 4, y + 3)
      }

      // Everything time-based gets clipped to the plot column so it scrolls cleanly
      // underneath the fixed "now" line instead of bleeding into the axis gutters.
      ctx.save()
      ctx.beginPath()
      ctx.rect(pad.l, 0, plotW, height)
      ctx.clip()

      // Time axis — relative gridlines/labels ("-10s", "now", "+10s"…) scrolling with data
      const tickStep = niceTickStep(halfWindow)
      ctx.font = '9.5px Inter, sans-serif'
      ctx.textAlign = 'center'
      for (let offset = 0; offset <= halfWindow; offset += tickStep) {
        for (const o of offset === 0 ? [0] : [-offset, offset]) {
          const x = xScale(playheadSec + o)
          ctx.strokeStyle = cssVar('--chart-grid')
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.moveTo(x, pad.t)
          ctx.lineTo(x, pad.t + plotH)
          ctx.stroke()
          ctx.fillStyle = cssVar('--text-muted')
          ctx.fillText(o === 0 ? 'now' : `${o > 0 ? '+' : ''}${o}s`, x, pad.t + plotH + 16)
        }
      }
      ctx.textAlign = 'left'

      // Ground truth regions (rounded, with border + label chip)
      const drawRegion = (
        r: ModelRegion,
        fill: string,
        border: string,
        label: string,
        labelAbove: boolean,
      ) => {
        const x0 = xScale(r.start)
        const x1 = xScale(r.end)
        const w = Math.max(2, x1 - x0)
        ctx.fillStyle = fill
        ctx.fillRect(x0, pad.t, w, plotH)
        ctx.strokeStyle = border
        ctx.lineWidth = 1.5
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(x0, pad.t)
        ctx.lineTo(x0, pad.t + plotH)
        ctx.moveTo(x1, pad.t)
        ctx.lineTo(x1, pad.t + plotH)
        ctx.stroke()
        ctx.setLineDash([])

        if (label) {
          ctx.font = '9.5px Inter, sans-serif'
          const tw = ctx.measureText(label).width
          const chipX = x0 + 3
          const chipY = labelAbove ? pad.t - 20 : pad.t + 4
          ctx.fillStyle = border
          roundRect(ctx, chipX, chipY, tw + 8, 14, 4)
          ctx.fill()
          ctx.fillStyle = '#05070d'
          ctx.fillText(label, chipX + 4, chipY + 10)
        }
      }

      for (const mem of memoryRegions) {
        const { fill, border } = regionFill(mem, playheadSec, false)
        drawRegion(mem, fill, border, mem.label, true)
      }

      for (const gt of groundTruthRegions) {
        const overlapMr = modelRegions.find((mr) => overlaps(mr, gt))
        if (overlapMr) continue
        const { fill, border } = regionFill(gt, playheadSec, false)
        drawRegion(gt, fill, border, gt.label, true)
      }

      for (const mr of modelRegions) {
        const hasGt = groundTruthRegions.some((gt) => overlaps(mr, gt))
        if (hasGt) continue
        const { fill, border } = regionFill(mr, playheadSec, false)
        drawRegion(mr, fill, border, mr.label, false)
      }

      for (const mr of modelRegions) {
        for (const gt of groundTruthRegions) {
          if (!overlaps(mr, gt)) continue
          const start = Math.max(mr.start, gt.start)
          const end = Math.min(mr.end, gt.end)
          const merged: ModelRegion = { ...mr, start, end }
          const { fill, border } = regionFill(merged, playheadSec, true)
          drawRegion(merged, fill, border, `${mr.label} / ${gt.label}`, true)
        }
      }

      const drawLine = (points: SeriesPoint[], stroke: string, width: number, glow?: string) => {
        if (points.length === 0) return
        if (points.length === 1) {
          const x = xScale(points[0].t)
          const y = yScale(points[0].y)
          ctx.beginPath()
          ctx.arc(x, y, 3, 0, Math.PI * 2)
          ctx.fillStyle = stroke
          if (glow) {
            ctx.shadowColor = glow
            ctx.shadowBlur = 8
          }
          ctx.fill()
          ctx.shadowBlur = 0
          return
        }
        ctx.beginPath()
        points.forEach((p, i) => {
          const x = xScale(p.t)
          const y = yScale(p.y)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        })
        ctx.strokeStyle = stroke
        ctx.lineWidth = width
        ctx.lineJoin = 'round'
        if (glow) {
          ctx.shadowColor = glow
          ctx.shadowBlur = 10
        }
        ctx.stroke()
        ctx.shadowBlur = 0
      }

      const strokeSeries = (
        points: SeriesPoint[],
        stroke: string,
        width: number,
        opts?: { dash?: number[]; alpha?: number; glow?: string },
      ) => {
        if (points.length < 2) {
          drawLine(points, stroke, width, opts?.glow)
          return
        }
        ctx.beginPath()
        points.forEach((p, i) => {
          const x = xScale(p.t)
          const y = yScale(p.y)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        })
        ctx.strokeStyle = stroke
        ctx.lineWidth = width
        ctx.lineJoin = 'round'
        ctx.globalAlpha = opts?.alpha ?? 1
        if (opts?.dash) ctx.setLineDash(opts.dash)
        if (opts?.glow) {
          ctx.shadowColor = opts.glow
          ctx.shadowBlur = 10
        }
        ctx.stroke()
        ctx.setLineDash([])
        ctx.shadowBlur = 0
        ctx.globalAlpha = 1
      }

      const px = pad.l + plotW / 2

      // Actual network — white solid through and past "now" (includes scheduled GT future)
      const pastForFill = fullActual.filter((p) => p.t <= playheadSec + 0.02)
      if (pastForFill.length > 1) {
        ctx.save()
        ctx.beginPath()
        ctx.rect(pad.l, pad.t, plotW / 2, plotH)
        ctx.clip()
        const fillGrad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH)
        fillGrad.addColorStop(0, cssVar('--chart-fill-top'))
        fillGrad.addColorStop(1, cssVar('--chart-fill-bottom'))
        ctx.beginPath()
        ctx.moveTo(xScale(pastForFill[0].t), pad.t + plotH)
        pastForFill.forEach((p) => ctx.lineTo(xScale(p.t), yScale(p.y)))
        ctx.lineTo(xScale(pastForFill[pastForFill.length - 1].t), pad.t + plotH)
        ctx.closePath()
        ctx.fillStyle = fillGrad
        ctx.fill()
        ctx.restore()
      }

      if (fullActual.length > 1) {
        strokeSeries(fullActual, cssVar('--chart-actual'), 2.25)
      } else {
        drawLine(fullActual, cssVar('--chart-actual'), 2.25)
      }

      // Model track — solid retrospective, dashed forecast (drawn on top of actual).
      const modelColor = cssVar('--chart-predicted')
      const modelGlow = cssVar('--chart-predicted-glow')

      if (pastModel.length > 1) {
        strokeSeries(pastModel, modelColor, 2.5, { glow: modelGlow })
      } else if (pastModel.length === 1) {
        drawLine(pastModel, modelColor, 2.5, modelGlow)
      }

      const yAtNow =
        pastModel.at(-1)?.y ??
        futureForecast[0]?.y ??
        pastActual.at(-1)?.y ??
        0
      const forecastLine: SeriesPoint[] =
        futureForecast.length > 0
          ? [{ t: playheadSec, y: yAtNow }, ...futureForecast]
          : []

      if (forecastLine.length > 1) {
        strokeSeries(forecastLine, modelColor, 2.75, {
          dash: [7, 5],
          glow: modelGlow,
        })
      }

      // Playhead — fixed dead-center dotted vertical + marker dots top & bottom.
      ctx.strokeStyle = cssVar('--playhead')
      ctx.lineWidth = 1.25
      ctx.setLineDash([4, 4])
      ctx.globalAlpha = 0.85
      ctx.beginPath()
      ctx.moveTo(px, pad.t)
      ctx.lineTo(px, pad.t + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1

      for (const cy of [pad.t, pad.t + plotH]) {
        ctx.beginPath()
        ctx.arc(px, cy, 3.5, 0, Math.PI * 2)
        ctx.fillStyle = cssVar('--playhead')
        ctx.shadowColor = cssVar('--playhead')
        ctx.shadowBlur = 6
        ctx.fill()
        ctx.shadowBlur = 0
      }

      ctx.restore()

      // Accuracy HUD chip bottom-right
      if (accuracy) {
        const hud = `MAE ${accuracy.lineMae.toFixed(3)}  ·  P ${(accuracy.regionPrecision * 100).toFixed(0)}%  ·  R ${(accuracy.regionRecall * 100).toFixed(0)}%`
        ctx.font = '600 10.5px Inter, sans-serif'
        const tw = ctx.measureText(hud).width
        const chipW = tw + 16
        const chipH = 20
        const chipX = pad.l + plotW - chipW
        const chipY = pad.t + plotH - chipH - 4
        ctx.fillStyle = cssVar('--bg-elevated')
        ctx.globalAlpha = 0.92
        roundRect(ctx, chipX, chipY, chipW, chipH, 8)
        ctx.fill()
        ctx.globalAlpha = 1
        ctx.fillStyle = cssVar('--text-secondary')
        ctx.fillText(hud, chipX + 8, chipY + 14)
      }
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(container)
    return () => ro.disconnect()
  }, [actual, observed, predicted, modelRegions, groundTruthRegions, memoryRegions, playheadSec, windowSec, dataStepSec, accuracy, height])

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-xl border border-[var(--border)]"
      style={{ background: 'var(--bg-surface-2)' }}
    >
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  )
}
