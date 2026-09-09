export interface ContainerInfo {
  name: string
  status: 'running' | 'stopped' | 'error'
  image?: string
}

export interface TrainingInfo {
  active: boolean
  job?: string
  progress?: number
  message?: string
}

export interface DatasetInfo {
  name: string
  path: string
  samples?: number
  sizeMb?: number
}

export interface DeviceInfo {
  cuda: boolean
  deviceName?: string
  vramGb?: number
}

export interface NetworkInfo {
  name: string
  cidr: string
  protected: boolean
}

export interface ModelSummary {
  id: string
  name: string
  version: string
  loaded: boolean
}

export interface MetricsPoint {
  t: number
  flowRate: number
  pAttack: number
}

export interface DashboardState {
  containers: ContainerInfo[]
  training: TrainingInfo
  datasets: DatasetInfo[]
  device: DeviceInfo
  networks: NetworkInfo[]
  models: ModelSummary[]
  metricsTimeseries: MetricsPoint[]
}
