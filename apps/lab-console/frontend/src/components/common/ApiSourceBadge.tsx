import { Badge } from './Badge'

interface ApiSourceBadgeProps {
  source: 'api' | 'mock'
}

export function ApiSourceBadge({ source }: ApiSourceBadgeProps) {
  return (
    <Badge tone={source === 'api' ? 'ok' : 'warn'} dot>
      {source === 'api' ? 'API' : 'mock'}
    </Badge>
  )
}
