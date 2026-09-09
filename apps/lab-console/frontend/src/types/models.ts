export interface ModelRegistryEntry {
  id: string
  name: string
  version: string
  checkpointPath: string
  sizeMb: number
  parameters: number
  classes: string[]
  featureDim: number
  seqLen: number
  lastTrained?: string
  tags: string[]
}
