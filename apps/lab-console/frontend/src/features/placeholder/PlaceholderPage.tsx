import { Construction } from 'lucide-react'
import { TopBar } from '../../app/TopBar'
import { IconTile } from '../../components/common/IconTile'

interface PlaceholderPageProps {
  title: string
  description: string
}

export function PlaceholderPage({ title, description }: PlaceholderPageProps) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <TopBar title={title} subtitle="Coming in phase 2" />
      <div className="flex flex-1 items-center justify-center p-12">
        <div className="glass-panel flex max-w-md flex-col items-center gap-4 rounded-2xl p-10 text-center">
          <IconTile icon={Construction} tone="warn" size={52} />
          <div>
            <h2 className="text-lg font-semibold text-[var(--text-primary)]">{title}</h2>
            <p className="mt-2 text-sm text-[var(--text-muted)]">{description}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
