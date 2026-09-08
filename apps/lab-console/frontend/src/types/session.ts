export type PolicyMode = 'ids' | 'ips'
export type SessionMode = 'live' | 'recorded'
export type LayoutMode = 'charts-only' | 'split' | 'terminal-focus'
export type RegionKind = 'suspicious' | 'attack' | 'ground_truth' | 'episodic'

export interface SeriesPoint {
  t: number
  y: number
}

export interface ModelRegion {
  start: number
  end: number
  kind: RegionKind
  label: string
  classId?: string
  resolved?: boolean
  correct?: boolean
}

export interface ModelAccuracy {
  lineMae: number
  regionPrecision: number
  regionRecall: number
}

export interface ModelSlot {
  id: string
  name: string
  predicted: SeriesPoint[]
  regions: ModelRegion[]
  accuracy: ModelAccuracy
}

export interface IpsStatus {
  armed: boolean
  blocker?: string
  blockAtSec?: number
  streaks: Record<string, number>
}

export interface SessionState {
  id: string
  playheadSec: number
  /** Wall-clock elapsed since scorer start (live only). */
  elapsedSec?: number
  /** Scorer capture window length in seconds (live only). */
  windowSec?: number
  durationSec: number
  mode: SessionMode
  policyMode: PolicyMode
  ips: IpsStatus
  actual: SeriesPoint[]
  models: ModelSlot[]
  groundTruthRegions: ModelRegion[]
  memoryRegions?: ModelRegion[]
}

export interface LogLine {
  ts: number
  kind: 'info' | 'phase' | 'alert' | 'block' | 'scorer' | 'attack'
  text: string
}

export interface ScriptDef {
  id: string
  label: string
  description: string
}
