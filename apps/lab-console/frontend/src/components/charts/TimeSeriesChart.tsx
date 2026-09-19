import { useEffect, useRef } from 'react'
import type {
  ModelAccuracy,
  ModelRegion,
  SeriesPoint,
} from '../../types/session'

export interface TimeSeriesChartProps {
  actual: SeriesPoint[]

  /** Per-model retrospective scored line (left of playhead). */
  observed?: SeriesPoint[]

  predicted: SeriesPoint[]

  modelRegions: ModelRegion[]

  groundTruthRegions: ModelRegion[]

  memoryRegions?: ModelRegion[]

  playheadSec: number

  /**
   * Visible time span (seconds), centered on playheadSec.
   * The "now" line stays fixed in the middle of the chart
   * and the data scrolls underneath it, like a live monitor.
   */
  windowSec?: number

  /**
   * Scorer window length — used to step future actual
   * padding in live mode.
   */
  dataStepSec?: number

  accuracy?: ModelAccuracy

  height?: number
}

/**
 * Pick a "nice" tick step so we get roughly
 * 2 gridlines on each side of "now".
 */
function niceTickStep(
  halfWindow: number,
): number {
  const target = halfWindow / 2

  const candidates = [
    1,
    2,
    5,
    10,
    15,
    20,
    30,
    60,
    120,
    300,
  ]

  let best = candidates[0]

  for (const c of candidates) {
    if (c <= target) {
      best = c
    }
  }

  return best
}

function cssVar(
  name: string,
): string {
  return getComputedStyle(
    document.documentElement,
  )
    .getPropertyValue(name)
    .trim()
}

/**
 * ============================================================
 * REGION COLOR SYSTEM
 * ============================================================
 *
 * Event colors are determined from the region label.
 *
 * T1046 / recon / scan
 *      -> Blue
 *
 * T1190 / enum / exploit
 *      -> Purple
 *
 * T1110 / bruteforce / spray
 *      -> Red
 *
 * Hue answers: "WHAT attack/event is this?"
 *
 * Fill intensity answers: "WHAT is the detection state?"
 *
 * Border style reinforces the state:
 *   actual   -> solid
 *   model    -> dashed
 *   overlap  -> strong solid
 *
 * The mapping is shared by ALL models.
 * Therefore Shaun v3 and HX-C show the same color for
 * the same attack/event type.
 *
 * We intentionally do NOT use a generic pink overlap
 * color.
 * ============================================================
 */
type RegionVisualRole =
  | 'actual'
  | 'model'
  | 'overlap'

function regionFill(
  region: ModelRegion,
  playhead: number,
  role: RegionVisualRole = 'actual',
): {
  fill: string
  border: string
} {
  const label =
    region.label.toLowerCase()

  /*
   * ==========================================================
   * EVENT COLOR SYSTEM
   *
   * Each attack type has one semantic hue:
   *
   *   Recon       -> Blue
   *   Enumeration -> Purple
   *   Brute force -> Red
   *
   * The role changes the intensity of that hue:
   *
   *   actual   -> base attack shade
   *   model    -> lighter prediction shade
   *   overlap  -> stronger blended shade
   *
   * This makes the chart communicate both:
   *   1. WHAT type of attack it is
   *   2. WHETHER the model prediction overlaps it
   *
   * The same mapping is shared by Shaun v3 and HX-C.
   * ==========================================================
   */

  if (
    label.includes('recon') ||
    label.includes('scan') ||
    label.includes('t1046')
  ) {
    if (role === 'model') {
      return {
        fill: 'rgba(96, 165, 250, 0.07)',
        border: '#60A5FA',
      }
    }

    if (role === 'overlap') {
      return {
        fill: 'rgba(59, 130, 246, 0.24)',
        border: '#2563EB',
      }
    }

    return {
      fill: 'rgba(59, 130, 246, 0.11)',
      border: '#3B82F6',
    }
  }

  if (
    label.includes('enum') ||
    label.includes('enumeration') ||
    label.includes('exploit') ||
    label.includes('t1190')
  ) {
    if (role === 'model') {
      return {
        fill: 'rgba(192, 132, 252, 0.07)',
        border: '#C084FC',
      }
    }

    if (role === 'overlap') {
      return {
        fill: 'rgba(168, 85, 247, 0.24)',
        border: '#9333EA',
      }
    }

    return {
      fill: 'rgba(168, 85, 247, 0.11)',
      border: '#A855F7',
    }
  }

  if (
    label.includes('spray') ||
    label.includes('bruteforce') ||
    label.includes('brute force') ||
    label.includes('password') ||
    label.includes('t1110')
  ) {
    if (role === 'model') {
      return {
        fill: 'rgba(248, 113, 113, 0.07)',
        border: '#F87171',
      }
    }

    if (role === 'overlap') {
      return {
        fill: 'rgba(239, 68, 68, 0.25)',
        border: '#DC2626',
      }
    }

    return {
      fill: 'rgba(239, 68, 68, 0.11)',
      border: '#EF4444',
    }
  }

  /*
   * Memory / episodic remains neutral.
   */
  if (
    region.kind === 'episodic'
  ) {
    return {
      fill: 'rgba(148, 163, 184, 0.04)',
      border: '#94A3B8',
    }
  }

  /*
   * Ground truth remains neutral because it is the
   * reference signal, not a prediction.
   */
  if (
    region.kind === 'ground_truth'
  ) {
    return {
      fill: 'rgba(255, 255, 255, 0.04)',
      border: '#CBD5E1',
    }
  }

  const past =
    region.end <= playhead

  /*
   * Generic suspicious regions.
   */
  if (
    region.kind === 'suspicious'
  ) {
    if (
      past &&
      region.resolved &&
      region.correct === false
    ) {
      return {
        fill: 'rgba(34, 211, 238, 0.05)',
        border: '#22D3EE',
      }
    }

    if (role === 'model') {
      return {
        fill: 'rgba(251, 191, 36, 0.06)',
        border: '#FBBF24',
      }
    }

    if (role === 'overlap') {
      return {
        fill: 'rgba(245, 158, 11, 0.22)',
        border: '#D97706',
      }
    }

    return {
      fill: 'rgba(245, 158, 11, 0.10)',
      border: '#F59E0B',
    }
  }

  /*
   * Generic attack regions.
   */
  if (
    region.kind === 'attack'
  ) {
    if (
      past &&
      region.resolved &&
      region.correct === false
    ) {
      return {
        fill: 'rgba(96, 165, 250, 0.05)',
        border: '#60A5FA',
      }
    }

    if (role === 'model') {
      return {
        fill: 'rgba(251, 113, 133, 0.045)',
        border: '#FB7185',
      }
    }

    if (role === 'overlap') {
      return {
        fill: 'rgba(244, 63, 94, 0.16)',
        border: '#E11D48',
      }
    }

    return {
      fill: 'rgba(244, 63, 94, 0.07)',
      border: '#F43F5E',
    }
  }

  return {
    fill: 'transparent',
    border: 'transparent',
  }
}

function overlaps(
  a: ModelRegion,
  b: ModelRegion,
): boolean {
  return (
    a.start < b.end &&
    b.start < a.end
  )
}

/**
 * UI-only attack family matching.
 *
 * This is used only to decide whether a model region and a
 * ground-truth region should receive the "BOTH" visual state.
 * No model/backend data is changed.
 */
function regionFamily(
  label: string,
): 'recon' | 'enum' | 'bruteforce' | 'suspicious' | 'other' {
  const normalized =
    label.toLowerCase()

  if (
    normalized.includes('t1046') ||
    normalized.includes('recon') ||
    normalized.includes('scan')
  ) {
    return 'recon'
  }

  if (
    normalized.includes('t1190') ||
    normalized.includes('enum') ||
    normalized.includes('enumeration') ||
    normalized.includes('exploit')
  ) {
    return 'enum'
  }

  if (
    normalized.includes('t1110') ||
    normalized.includes('spray') ||
    normalized.includes('bruteforce') ||
    normalized.includes('brute force') ||
    normalized.includes('password')
  ) {
    return 'bruteforce'
  }

  if (
    normalized.includes('suspicious') ||
    normalized.includes('anomaly')
  ) {
    return 'suspicious'
  }

  return 'other'
}

function sameAttackFamily(
  a: ModelRegion,
  b: ModelRegion,
): boolean {
  const aFamily =
    regionFamily(a.label)
  const bFamily =
    regionFamily(b.label)

  return (
    aFamily !== 'other' &&
    aFamily === bFamily
  )
}

/**
 * Convert technical attack labels into short,
 * readable labels for the chart.
 *
 * This changes only the DISPLAYED label.
 * The underlying model region data is untouched.
 */
function formatEventLabel(
  label: string,
): string {
  const normalized =
    label.toLowerCase()

  if (
    normalized.includes('t1046') ||
    normalized.includes('recon') ||
    normalized.includes('scan')
  ) {
    return 'RECON · T1046'
  }

  if (
    normalized.includes('t1190') ||
    normalized.includes('enum') ||
    normalized.includes('enumeration') ||
    normalized.includes('exploit')
  ) {
    return 'ENUM · T1190'
  }

  if (
    normalized.includes('t1110') ||
    normalized.includes('spray') ||
    normalized.includes('bruteforce') ||
    normalized.includes('brute force') ||
    normalized.includes('password')
  ) {
    return 'BRUTE FORCE · T1110'
  }

  return label
}

/**
 * Observed past + flat continuation through the visible
 * window (attacks shown as region bands).
 */
function buildActualLine(
  actual: SeriesPoint[],
  playheadSec: number,
  minT: number,
  maxT: number,
  stepSec: number,
): SeriesPoint[] {
  const step = Math.max(
    stepSec * 0.5,
    0.25,
  )

  const past = actual
    .filter(
      (p) =>
        p.t <=
        playheadSec + 0.02,
    )
    .sort(
      (a, b) =>
        a.t - b.t,
    )

  const out: SeriesPoint[] = []

  for (const p of past) {
    if (
      out.length &&
      Math.abs(
        out[out.length - 1].t -
          p.t,
      ) <
        step * 0.2
    ) {
      continue
    }

    out.push(p)
  }

  const baseline =
    out.at(-1)?.y ?? 0.06

  const lastT =
    out.at(-1)?.t

  if (
    lastT !== undefined &&
    lastT <
      playheadSec - 0.01
  ) {
    out.push({
      t: playheadSec,
      y: baseline,
    })
  }

  let t = Math.max(
    out.at(-1)?.t ??
      playheadSec,
    playheadSec,
  )

  while (
    t <
      maxT - 0.01
  ) {
    t =
      Math.round(
        (t + step) * 1000,
      ) / 1000

    if (
      out.some(
        (p) =>
          Math.abs(
            p.t - t,
          ) <
            step * 0.35,
      )
    ) {
      continue
    }

    out.push({
      t,
      y: baseline,
    })
  }

  return out.filter(
    (p) =>
      p.t >=
        minT - step &&
      p.t <=
        maxT + step,
  )
}

function modelPastPoints(
  observed: SeriesPoint[],
  predicted: SeriesPoint[],
  playheadSec: number,
): SeriesPoint[] {
  if (
    observed.length > 0
  ) {
    return observed
      .filter(
        (p) =>
          p.t <=
          playheadSec + 0.05,
      )
      .sort(
        (a, b) =>
          a.t - b.t,
      )
  }

  return predicted
    .filter(
      (p) =>
        p.t <=
        playheadSec + 0.05,
    )
    .sort(
      (a, b) =>
        a.t - b.t,
    )
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(
    r,
    Math.abs(w) / 2,
    h / 2,
  )

  ctx.beginPath()

  ctx.moveTo(
    x + rr,
    y,
  )

  ctx.arcTo(
    x + w,
    y,
    x + w,
    y + h,
    rr,
  )

  ctx.arcTo(
    x + w,
    y + h,
    x,
    y + h,
    rr,
  )

  ctx.arcTo(
    x,
    y + h,
    x,
    y,
    rr,
  )

  ctx.arcTo(
    x,
    y,
    x + w,
    y,
    rr,
  )

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
  const canvasRef =
    useRef<HTMLCanvasElement>(
      null,
    )

  const containerRef =
    useRef<HTMLDivElement>(
      null,
    )

  useEffect(() => {
    const canvas =
      canvasRef.current

    const container =
      containerRef.current

    if (
      !canvas ||
      !container
    ) {
      return
    }

    const draw = () => {
      const dpr =
        window.devicePixelRatio ||
        1

      const width =
        container.clientWidth

      canvas.width =
        width * dpr

      canvas.height =
        height * dpr

      canvas.style.width =
        `${width}px`

      canvas.style.height =
        `${height}px`

      const ctx =
        canvas.getContext(
          '2d',
        )

      if (!ctx) {
        return
      }

      ctx.scale(
        dpr,
        dpr,
      )

      ctx.clearRect(
        0,
        0,
        width,
        height,
      )

      const pad = {
        l: 40,
        r: 14,
        t: 30,
        b: 26,
      }

      const plotW =
        width -
        pad.l -
        pad.r

      const plotH =
        height -
        pad.t -
        pad.b

      /*
       * ========================================================
       * TIME WINDOW
       * ========================================================
       *
       * "Now" is always dead-center.
       * The window scrolls underneath the fixed playhead.
       */
      const halfWindow =
        windowSec / 2

      const minT =
        playheadSec -
        halfWindow

      const maxT =
        playheadSec +
        halfWindow

      const pastModel =
        modelPastPoints(
          observed,
          predicted,
          playheadSec,
        ).filter(
          (p) =>
            p.t >=
            minT - 0.5,
        )

      const futureForecast =
        predicted
          .filter(
            (p) =>
              p.t >
                playheadSec +
                  0.001 &&
              p.t <=
                maxT + 0.5,
          )
          .sort(
            (a, b) =>
              a.t - b.t,
          )

      const fullActual =
        buildActualLine(
          actual,
          playheadSec,
          minT,
          maxT,
          dataStepSec,
        )

      const pastActual =
        fullActual.filter(
          (p) =>
            p.t <=
            playheadSec + 0.02,
        )

      /*
       * Scale from model lines + measured actual.
       * Synthetic GT future spikes are not used.
       */
      const scaleYs = [
        ...pastModel.map(
          (p) => p.y,
        ),

        ...futureForecast.map(
          (p) => p.y,
        ),

        ...pastActual.map(
          (p) => p.y,
        ),
      ]

      const minY =
        scaleYs.length
          ? Math.min(
              ...scaleYs,
              0,
            ) - 0.05
          : -0.05

      const maxY =
        scaleYs.length
          ? Math.max(
              ...scaleYs,
              1,
            ) + 0.05
          : 1.05

      const xScale = (
        t: number,
      ) =>
        pad.l +
        ((t - minT) /
          (maxT - minT || 1)) *
          plotW

      const yScale = (
        y: number,
      ) =>
        pad.t +
        plotH -
        ((y - minY) /
          (maxY - minY || 1)) *
          plotH

      /*
       * ========================================================
       * BACKGROUND GRID
       * ========================================================
       */

      ctx.strokeStyle =
        cssVar(
          '--chart-grid',
        )

      ctx.lineWidth = 1

      ctx.font =
        '10px Inter, sans-serif'

      ctx.fillStyle =
        cssVar(
          '--text-muted',
        )

      for (
        let i = 0;
        i <= 3;
        i++
      ) {
        const y =
          pad.t +
          (plotH * i) /
            3

        ctx.beginPath()

        ctx.moveTo(
          pad.l,
          y,
        )

        ctx.lineTo(
          pad.l + plotW,
          y,
        )

        ctx.stroke()

        const val =
          maxY -
          ((maxY - minY) *
            i) /
            3

        ctx.fillText(
          val.toFixed(2),
          4,
          y + 3,
        )
      }

      /*
       * Everything time-based gets clipped to the plot area.
       */
      ctx.save()

      ctx.beginPath()

      ctx.rect(
        pad.l,
        0,
        plotW,
        height,
      )

      ctx.clip()

      /*
       * ========================================================
       * STEP 4 — PAST / NOW / FORECAST
       * ========================================================
       *
       * The right half of the chart is the model's forecast
       * horizon. The tint is intentionally subtle so event
       * regions and data lines remain the main focus.
       */
      const px =
        pad.l +
        plotW / 2

      const forecastX =
        px

      const forecastW =
        Math.max(
          0,
          pad.l +
            plotW -
            forecastX,
        )

      ctx.fillStyle =
        'rgba(34, 211, 238, 0.035)'

      ctx.fillRect(
        forecastX,
        pad.t,
        forecastW,
        plotH,
      )

      /*
       * Forecast-side edge.
       * This gives the user an immediate visual transition
       * without adding another heavy card or border.
       */
      ctx.strokeStyle =
        'rgba(34, 211, 238, 0.16)'

      ctx.lineWidth = 1

      ctx.beginPath()

      ctx.moveTo(
        forecastX,
        pad.t,
      )

      ctx.lineTo(
        forecastX,
        pad.t + plotH,
      )

      ctx.stroke()

      /*
       * Section labels.
       */
      ctx.font =
        '600 8.5px Inter, sans-serif'

      ctx.textBaseline =
        'middle'

      ctx.letterSpacing = '0px'

      ctx.fillStyle =
        'rgba(148, 163, 184, 0.72)'

      ctx.textAlign =
        'left'

      ctx.fillText(
        'PAST',
        pad.l + 8,
        pad.t - 15,
      )

      ctx.fillStyle =
        'rgba(34, 211, 238, 0.9)'

      ctx.textAlign =
        'left'

      ctx.fillText(
        'FORECAST',
        forecastX + 8,
        pad.t - 15,
      )

      /*
       * ========================================================
       * TIME AXIS
       * ========================================================
       */

      const tickStep =
        niceTickStep(
          halfWindow,
        )

      ctx.font =
        '9.5px Inter, sans-serif'

      ctx.textAlign =
        'center'

      ctx.textBaseline =
        'alphabetic'

      for (
        let offset = 0;
        offset <= halfWindow;
        offset += tickStep
      ) {
        for (
          const o of
            offset === 0
              ? [0]
              : [
                  -offset,
                  offset,
                ]
        ) {
          const x =
            xScale(
              playheadSec +
                o,
            )

          ctx.strokeStyle =
            cssVar(
              '--chart-grid',
            )

          ctx.lineWidth = 1

          ctx.beginPath()

          ctx.moveTo(
            x,
            pad.t,
          )

          ctx.lineTo(
            x,
            pad.t + plotH,
          )

          ctx.stroke()

          ctx.fillStyle =
            cssVar(
              '--text-muted',
            )

          ctx.fillText(
            o === 0
              ? 'now'
              : `${
                  o > 0
                    ? '+'
                    : ''
                }${o}s`,
            x,
            pad.t +
              plotH +
              16,
          )
        }
      }

      ctx.textAlign =
        'left'

      /*
       * ========================================================
       * REGION DRAWER
       * ========================================================
       */

      type RegionBorderStyle =
        | 'solid'
        | 'dashed'
        | 'strong'

      const drawRegion = (
        r: ModelRegion,
        fill: string,
        border: string,
        label: string,
        labelAbove: boolean,
        borderStyle: RegionBorderStyle,
        statusLabel: string,
      ) => {
        const x0 =
          xScale(r.start)

        const x1 =
          xScale(r.end)

        const w = Math.max(
          2,
          x1 - x0,
        )

        /*
         * ========================================================
         * REGION FILL
         * ========================================================
         *
         * Fill intensity communicates detection state:
         *
         *   light  -> model prediction
         *   medium -> actual
         *   strong -> both
         *
         * Hue still communicates attack type.
         */
        ctx.fillStyle =
          fill

        ctx.fillRect(
          x0,
          pad.t,
          w,
          plotH,
        )

        /*
         * ========================================================
         * REGION BOUNDARY
         * ========================================================
         *
         * Border style is a second, non-color cue:
         *
         *   dashed -> model prediction
         *   solid  -> actual
         *   strong -> actual + model prediction
         */
        ctx.strokeStyle =
          border

        ctx.lineWidth =
          borderStyle === 'strong'
            ? 2.5
            : 1.8

        if (
          borderStyle === 'dashed'
        ) {
          ctx.setLineDash([
            5,
            4,
          ])
        } else {
          ctx.setLineDash([])
        }

        ctx.beginPath()

        ctx.moveTo(
          x0,
          pad.t,
        )

        ctx.lineTo(
          x0,
          pad.t + plotH,
        )

        ctx.moveTo(
          x1,
          pad.t,
        )

        ctx.lineTo(
          x1,
          pad.t + plotH,
        )

        ctx.stroke()

        ctx.setLineDash([])

        /*
         * ========================================================
         * EVENT LABEL
         * ========================================================
         *
         * The compact state prefix makes the meaning explicit:
         *
         *   PREDICTED -> model only
         *   ACTUAL    -> ground truth only
         *   BOTH      -> model + ground truth
         *
         * This prevents the admin from having to infer the
         * state from color alone.
         */
        if (label) {
          const fullLabel =
            `${statusLabel} · ${label}`

          ctx.font =
            '600 8.5px Inter, sans-serif'

          const tw =
            ctx.measureText(
              fullLabel,
            ).width

          const chipX =
            x0 + 3

          const chipY =
            labelAbove
              ? pad.t - 20
              : pad.t + 4

          const chipW =
            Math.min(
              tw + 8,
              Math.max(
                72,
                width - chipX - 4,
              ),
            )

          ctx.fillStyle =
            border

          roundRect(
            ctx,
            chipX,
            chipY,
            chipW,
            15,
            4,
          )

          ctx.fill()

          ctx.fillStyle =
            '#05070d'

          ctx.save()
          ctx.beginPath()
          ctx.rect(
            chipX,
            chipY,
            chipW,
            15,
          )
          ctx.clip()

          ctx.fillText(
            fullLabel,
            chipX + 4,
            chipY + 10.5,
          )

          ctx.restore()
        }
      }


      /*
       * ========================================================
       * MEMORY / EPISODIC REGIONS
       * ========================================================
       */

      for (
        const mem of
          memoryRegions
      ) {
        const {
          fill,
          border,
        } =
          regionFill(
            mem,
            playheadSec,
            'actual',
          )

        drawRegion(
          mem,
          fill,
          border,
          formatEventLabel(
            mem.label,
          ),
          true,
          'solid',
          'EVENT',
        )
      }

      /*
       * ========================================================
       * GROUND TRUTH
       * ========================================================
       *
       * Ground truth is the actual attack/event signal.
       * It uses the base shade for each attack type.
       * ========================================================
       */

      for (
        const gt of
          groundTruthRegions
      ) {
        const {
          fill,
          border,
        } =
          regionFill(
            gt,
            playheadSec,
            'actual',
          )

        drawRegion(
          gt,
          fill,
          border,
          formatEventLabel(
            gt.label,
          ),
          true,
          'solid',
          'ACTUAL',
        )
      }

      /*
       * ========================================================
       * MODEL REGIONS
       * ========================================================
       *
       * Model predictions use a lighter shade of the SAME
       * attack color and a dashed boundary:
       *
       *   Recon       -> lighter blue
       *   Enumeration -> lighter purple
       *   Brute force -> lighter red
       *
       * The overlap pass below strengthens the shared area.
       * ========================================================
       */

      for (
        const mr of
          modelRegions
      ) {
        const {
          fill,
          border,
        } =
          regionFill(
            mr,
            playheadSec,
            'model',
          )

        drawRegion(
          mr,
          fill,
          border,
          formatEventLabel(
            mr.label,
          ),
          false,
          'dashed',
          'PREDICTED',
        )
      }

      /*
       * ========================================================
       * MODEL + GROUND TRUTH OVERLAP
       * ========================================================
       *
       * A "BOTH" state is rendered only when:
       *
       *   1. the model region overlaps the ground-truth region, and
       *   2. both regions belong to the same known attack family.
       *
       * This is important: temporal overlap alone does NOT mean
       * the model predicted the same attack type.
       *
       * UI-only change:
       * no prediction, backend, API, or model data is modified.
       */
      for (
        const mr of
          modelRegions
      ) {
        for (
          const gt of
            groundTruthRegions
        ) {
          if (
            !overlaps(
              mr,
              gt,
            ) ||
            !sameAttackFamily(
              mr,
              gt,
            )
          ) {
            continue
          }

          const start =
            Math.max(
              mr.start,
              gt.start,
            )

          const end =
            Math.min(
              mr.end,
              gt.end,
            )

          if (end <= start) {
            continue
          }

          const overlapRegion: ModelRegion =
            {
              ...mr,
              start,
              end,
            }

          const {
            fill,
            border,
          } =
            regionFill(
              overlapRegion,
              playheadSec,
              'overlap',
            )

          drawRegion(
            overlapRegion,
            fill,
            border,
            formatEventLabel(
              mr.label,
            ),
            true,
            'strong',
            'BOTH',
          )
        }
      }

      /*
       * ========================================================
       * LINE DRAWING
       * ========================================================
       */

      const drawLine = (
        points: SeriesPoint[],
        stroke: string,
        width: number,
        glow?: string,
      ) => {
        if (
          points.length === 0
        ) {
          return
        }

        if (
          points.length === 1
        ) {
          const x =
            xScale(
              points[0].t,
            )

          const y =
            yScale(
              points[0].y,
            )

          ctx.beginPath()

          ctx.arc(
            x,
            y,
            3,
            0,
            Math.PI * 2,
          )

          ctx.fillStyle =
            stroke

          if (glow) {
            ctx.shadowColor =
              glow

            ctx.shadowBlur =
              8
          }

          ctx.fill()

          ctx.shadowBlur =
            0

          return
        }

        ctx.beginPath()

        points.forEach(
          (p, i) => {
            const x =
              xScale(p.t)

            const y =
              yScale(p.y)

            if (i === 0) {
              ctx.moveTo(
                x,
                y,
              )
            } else {
              ctx.lineTo(
                x,
                y,
              )
            }
          },
        )

        ctx.strokeStyle =
          stroke

        ctx.lineWidth =
          width

        ctx.lineJoin =
          'round'

        if (glow) {
          ctx.shadowColor =
            glow

          ctx.shadowBlur =
            10
        }

        ctx.stroke()

        ctx.shadowBlur =
          0
      }

      const strokeSeries = (
        points: SeriesPoint[],
        stroke: string,
        width: number,
        opts?: {
          dash?: number[]
          alpha?: number
          glow?: string
        },
      ) => {
        if (
          points.length < 2
        ) {
          drawLine(
            points,
            stroke,
            width,
            opts?.glow,
          )

          return
        }

        ctx.beginPath()

        points.forEach(
          (p, i) => {
            const x =
              xScale(p.t)

            const y =
              yScale(p.y)

            if (i === 0) {
              ctx.moveTo(
                x,
                y,
              )
            } else {
              ctx.lineTo(
                x,
                y,
              )
            }
          },
        )

        ctx.strokeStyle =
          stroke

        ctx.lineWidth =
          width

        ctx.lineJoin =
          'round'

        ctx.globalAlpha =
          opts?.alpha ?? 1

        if (opts?.dash) {
          ctx.setLineDash(
            opts.dash,
          )
        }

        if (opts?.glow) {
          ctx.shadowColor =
            opts.glow

          ctx.shadowBlur =
            10
        }

        ctx.stroke()

        ctx.setLineDash(
          [],
        )

        ctx.shadowBlur =
          0

        ctx.globalAlpha =
          1
      }

      /*
       * ========================================================
       * ACTUAL NETWORK
       *
       * White solid line.
       * ========================================================
       */

      const pastForFill =
        fullActual.filter(
          (p) =>
            p.t <=
            playheadSec + 0.02,
        )

      if (
        pastForFill.length > 1
      ) {
        ctx.save()

        ctx.beginPath()

        ctx.rect(
          pad.l,
          pad.t,
          plotW / 2,
          plotH,
        )

        ctx.clip()

        const fillGrad =
          ctx.createLinearGradient(
            0,
            pad.t,
            0,
            pad.t + plotH,
          )

        fillGrad.addColorStop(
          0,
          cssVar(
            '--chart-fill-top',
          ),
        )

        fillGrad.addColorStop(
          1,
          cssVar(
            '--chart-fill-bottom',
          ),
        )

        ctx.beginPath()

        ctx.moveTo(
          xScale(
            pastForFill[0].t,
          ),
          pad.t + plotH,
        )

        pastForFill.forEach(
          (p) =>
            ctx.lineTo(
              xScale(p.t),
              yScale(p.y),
            ),
        )

        ctx.lineTo(
          xScale(
            pastForFill[
              pastForFill.length -
                1
            ].t,
          ),
          pad.t + plotH,
        )

        ctx.closePath()

        ctx.fillStyle =
          fillGrad

        ctx.fill()

        ctx.restore()
      }

      if (
        fullActual.length > 1
      ) {
        strokeSeries(
          fullActual,
          cssVar(
            '--chart-actual',
          ),
          2.25,
        )
      } else {
        drawLine(
          fullActual,
          cssVar(
            '--chart-actual',
          ),
          2.25,
        )
      }

      /*
       * ========================================================
       * MODEL TRACK
       *
       * Solid retrospective.
       * Dashed forecast.
       * ========================================================
       */

      const modelColor =
        cssVar(
          '--chart-predicted',
        )

      const modelGlow =
        cssVar(
          '--chart-predicted-glow',
        )

      if (
        pastModel.length > 1
      ) {
        strokeSeries(
          pastModel,
          modelColor,
          2.5,
          {
            glow: modelGlow,
          },
        )
      } else if (
        pastModel.length === 1
      ) {
        drawLine(
          pastModel,
          modelColor,
          2.5,
          modelGlow,
        )
      }

      const yAtNow =
        pastModel.at(-1)?.y ??
        futureForecast[0]?.y ??
        pastActual.at(-1)?.y ??
        0

      const forecastLine:
        SeriesPoint[] =
        futureForecast.length > 0
          ? [
              {
                t: playheadSec,
                y: yAtNow,
              },
              ...futureForecast,
            ]
          : []

      if (
        forecastLine.length > 1
      ) {
        strokeSeries(
          forecastLine,
          modelColor,
          2.75,
          {
            dash: [
              7,
              5,
            ],
            glow: modelGlow,
          },
        )
      }

      /*
       * ========================================================
       * PLAYHEAD / NOW
       * ========================================================
       */

      ctx.strokeStyle =
        cssVar(
          '--playhead',
        )

      ctx.lineWidth =
        1.5

      ctx.setLineDash([
        4,
        4,
      ])

      ctx.globalAlpha =
        0.95

      ctx.beginPath()

      ctx.moveTo(
        px,
        pad.t,
      )

      ctx.lineTo(
        px,
        pad.t + plotH,
      )

      ctx.stroke()

      ctx.setLineDash(
        [],
      )

      ctx.globalAlpha =
        1

      /*
       * NOW marker.
       */
      const nowLabel =
        'NOW'

      ctx.font =
        '700 8.5px Inter, sans-serif'

      const nowWidth =
        ctx.measureText(
          nowLabel,
        ).width +
        12

      const nowX =
        px -
        nowWidth / 2

      const nowY =
        pad.t -
        23

      ctx.fillStyle =
        cssVar(
          '--playhead',
        )

      roundRect(
        ctx,
        nowX,
        nowY,
        nowWidth,
        15,
        4,
      )

      ctx.fill()

      ctx.fillStyle =
        '#05070d'

      ctx.textAlign =
        'center'

      ctx.textBaseline =
        'middle'

      ctx.fillText(
        nowLabel,
        px,
        nowY + 7.5,
      )

      ctx.textAlign =
        'left'

      ctx.textBaseline =
        'alphabetic'

      /*
       * Playhead endpoint markers.
       */
      for (
        const cy of [
          pad.t,
          pad.t + plotH,
        ]
      ) {
        ctx.beginPath()

        ctx.arc(
          px,
          cy,
          3.5,
          0,
          Math.PI * 2,
        )

        ctx.fillStyle =
          cssVar(
            '--playhead',
          )

        ctx.shadowColor =
          cssVar(
            '--playhead',
          )

        ctx.shadowBlur =
          6

        ctx.fill()

        ctx.shadowBlur =
          0
      }

      ctx.restore()

      /*
       * ========================================================
       * ACCURACY HUD
       * ========================================================
       */

      if (accuracy) {
        const hud =
          `MAE ${accuracy.lineMae.toFixed(3)}  ·  P ${(accuracy.regionPrecision * 100).toFixed(0)}%  ·  R ${(accuracy.regionRecall * 100).toFixed(0)}%`

        ctx.font =
          '600 10.5px Inter, sans-serif'

        const tw =
          ctx.measureText(
            hud,
          ).width

        const chipW =
          tw + 16

        const chipH =
          20

        const chipX =
          pad.l +
          plotW -
          chipW

        const chipY =
          pad.t +
          plotH -
          chipH -
          4

        ctx.fillStyle =
          cssVar(
            '--bg-elevated',
          )

        ctx.globalAlpha =
          0.92

        roundRect(
          ctx,
          chipX,
          chipY,
          chipW,
          chipH,
          8,
        )

        ctx.fill()

        ctx.globalAlpha =
          1

        ctx.fillStyle =
          cssVar(
            '--text-secondary',
          )

        ctx.fillText(
          hud,
          chipX + 8,
          chipY + 14,
        )
      }
    }

    draw()

    const ro =
      new ResizeObserver(
        draw,
      )

    ro.observe(
      container,
    )

    return () =>
      ro.disconnect()
  }, [
    actual,
    observed,
    predicted,
    modelRegions,
    groundTruthRegions,
    memoryRegions,
    playheadSec,
    windowSec,
    dataStepSec,
    accuracy,
    height,
  ])

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-xl border border-[var(--border)]"
      style={{
        background:
          'var(--bg-surface-2)',
      }}
    >
      <canvas
        ref={canvasRef}
        className="block w-full"
      />
    </div>
  )
}