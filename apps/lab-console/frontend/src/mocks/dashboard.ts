import type { DashboardState } from '../types/dashboard'

export const mockDashboard: DashboardState = {
  containers: [
    { name: 'harborline-site', status: 'running', image: 'prism/harborline:latest' },
    { name: 'attacker-bot', status: 'running', image: 'prism/attacker:latest' },
    { name: 'suricata', status: 'running', image: 'jasonish/suricata:latest' },
    { name: 'redteam', status: 'stopped', image: 'prism/redteam:latest' },
  ],
  training: {
    active: false,
    message: 'Idle — last lab-adapt finished 2h ago',
  },
  datasets: [
    { name: 'lab_events buffer', path: 'data/lab_events/forecast_adapt_buffer.npz', samples: 1842, sizeMb: 12.4 },
    { name: 'adversarial missed', path: 'data/raw/adversarial/missed/', samples: 37, sizeMb: 2.1 },
    { name: 'CIC-IDS val (5s)', path: 'data/processed/aryan_cic_5s/', samples: 42000, sizeMb: 890 },
  ],
  device: {
    cuda: true,
    deviceName: 'NVIDIA GeForce RTX 4070',
    vramGb: 12,
  },
  networks: [
    { name: 'lab_front', cidr: '172.28.0.0/24', protected: true },
    { name: 'lab_back', cidr: '172.29.0.0/24', protected: true },
    { name: 'internet_sim', cidr: '10.0.0.0/24', protected: false },
  ],
  models: [
    { id: 'shaun_v3', name: 'Shaun v3', version: 'w5s', loaded: true },
    { id: 'hx_c', name: 'HX-C', version: 'lab', loaded: true },
    { id: 'gen8_world_model', name: 'PRISM Gen 8', version: 'universal-5s', loaded: true },
    { id: 'xmt_01', name: 'XMT.01', version: 'v1', loaded: false },
  ],
  metricsTimeseries: Array.from({ length: 60 }, (_, i) => ({
    t: i,
    flowRate: 120 + Math.sin(i / 8) * 40 + (i > 35 ? 80 : 0),
    pAttack: Math.min(0.95, Math.max(0.02, i > 30 ? (i - 30) * 0.04 : 0.05 + Math.random() * 0.03)),
  })),
}
