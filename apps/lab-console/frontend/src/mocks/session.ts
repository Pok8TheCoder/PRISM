import type { LogLine, ModelRegion, ModelSlot, SessionState } from '../types/session'

function series(start: number, len: number, base: number, amp: number, phase = 0): { t: number; y: number }[] {
  return Array.from({ length: len }, (_, i) => ({
    t: start + i,
    y: base + Math.sin((i + phase) / 6) * amp + (i > len * 0.55 ? (i - len * 0.55) * 0.008 : 0),
  }))
}

const gtRegions: ModelRegion[] = [
  { start: 18, end: 28, kind: 'ground_truth', label: 'recon', classId: 'T1046' },
  { start: 30, end: 42, kind: 'ground_truth', label: 'enum', classId: 'T1190' },
  { start: 44, end: 55, kind: 'ground_truth', label: 'spray', classId: 'T1110' },
]

function modelRegions(offset: number): ModelRegion[] {
  return [
    { start: 19 + offset, end: 27 + offset, kind: 'suspicious', label: 'T1046_service_scan', classId: 'T1046' },
    { start: 32 + offset, end: 40 + offset, kind: 'attack', label: 'T1190_web_exploit', classId: 'T1190', resolved: true, correct: true },
    { start: 46 + offset, end: 52 + offset, kind: 'suspicious', label: 'T1110_ssh_bruteforce', classId: 'T1110', resolved: true, correct: false },
  ]
}

const models: ModelSlot[] = [
  {
    id: 'shaun_v3',
    name: 'Shaun v3',
    predicted: series(0, 70, 0.12, 0.04, 0),
    regions: modelRegions(0),
    accuracy: { lineMae: 0.042, regionPrecision: 0.78, regionRecall: 0.71 },
  },
  {
    id: 'hx_c',
    name: 'HX-C',
    predicted: series(0, 70, 0.1, 0.05, 2),
    regions: modelRegions(1),
    accuracy: { lineMae: 0.038, regionPrecision: 0.82, regionRecall: 0.75 },
  },
  {
    id: 'gen10_world_model',
    name: 'PRISM Gen 10 World Model',
    predicted: series(0, 70, 0.08, 0.03, 1),
    regions: modelRegions(0),
    accuracy: { lineMae: 0.018, regionPrecision: 0.992, regionRecall: 0.988 },
  },
]

export const mockSession: SessionState = {
  id: 'demo-001',
  playheadSec: 38,
  durationSec: 70,
  mode: 'recorded',
  policyMode: 'ips',
  ips: {
    armed: true,
    blocker: undefined,
    streaks: { shaun_v3: 1, hx_c: 0 },
  },
  actual: series(0, 70, 0.11, 0.035, 1),
  models,
  groundTruthRegions: gtRegions,
}

export const mockLogs: LogLine[] = [
  { ts: 1, kind: 'phase', text: 'Warming up on Harborline benign traffic…' },
  { ts: 2, kind: 'info', text: 'LIVE — scoring on; IPS arms for enum/spray/loot only.' },
  { ts: 18, kind: 'phase', text: '[auto-attack] phase=recon' },
  { ts: 30, kind: 'phase', text: '[auto-attack] phase=enum' },
  { ts: 32, kind: 'scorer', text: 'HX-C P(attack)=0.52 window=32' },
  { ts: 38, kind: 'alert', text: 'FORECAST ALERT by HX-C at t+38.0s' },
  { ts: 44, kind: 'phase', text: '[auto-attack] phase=spray' },
]

export const mockScripts = [
  { id: 'killchain-recon', label: 'Kill-chain: recon', description: 'Port scan phase' },
  { id: 'killchain-enum', label: 'Kill-chain: enum', description: 'Directory bust' },
  { id: 'killchain-spray', label: 'Kill-chain: spray', description: 'Credential spray' },
  { id: 'killchain-loot', label: 'Kill-chain: loot', description: 'Payroll download' },
  { id: 'killchain-all', label: 'Kill-chain: all', description: 'Full chain' },
  { id: 'auto-attack', label: 'Auto-attack', description: 'demo_forecast --auto-attack' },
  { id: 'lab-up', label: 'Lab up', description: 'lab_ctl up' },
  { id: 'lab-down', label: 'Lab down', description: 'lab_ctl down' },
  { id: 'bench-fair-ids', label: 'Fair IDS bench', description: 'Offline PCAP scoring' },
  { id: 'forecast-record', label: 'Record session', description: 'WebM + manifest' },
]
