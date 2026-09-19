import { Badge } from './Badge'

interface ApiSourceBadgeProps {
  source: 'api' | 'mock'
}

export function ApiSourceBadge({
  source,
}: ApiSourceBadgeProps) {
  const isApi = source === 'api'

  return (
    <Badge
      tone={isApi ? 'ok' : 'warn'}
      dot
      className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide"
    >
      {isApi ? 'API' : 'Mock'}
    </Badge>
  )
}
