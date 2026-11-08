import { ArrowRight, X } from 'lucide-react'
import { useData, useTwin } from '../state/store'
import { grapStage } from '../lib/aqi'
import { Button, PmValue } from '../ui/primitives'

export interface WardPick {
  code: string
  x: number
  y: number
}

export function WardPopover({ pick, pm, onClose }: { pick: WardPick; pm: number | null; onClose: () => void }) {
  const data = useData()
  const setView = useTwin((s) => s.setView)
  const ward = data.wards.wards.find((w) => w.code === pick.code)
  if (!ward) return null
  const g = grapStage(pm)
  const left = Math.min(pick.x + 12, window.innerWidth - 300)
  const top = Math.min(pick.y + 12, window.innerHeight - 380)

  return (
    <div className="panel absolute z-20 w-[272px] p-4" style={{ left, top }} role="dialog" aria-label={`Ward ${ward.code}`}>
      <button onClick={onClose} className="absolute right-2.5 top-2.5 rounded-md p-1 text-faint hover:text-paper" aria-label="Close">
        <X size={14} />
      </button>
      <div className="eyebrow">Ward {ward.code} · {ward.zone}</div>
      <div className="mt-0.5 pr-5 text-[15px] font-medium leading-snug">{ward.name}</div>
      <div className="mt-3 flex items-end justify-between">
        <PmValue value={pm} size="md" />
        <span
          className="rounded-full px-2 py-0.5 text-2xs font-medium"
          style={{ color: g.color, background: `${g.color}1f`, boxShadow: `inset 0 0 0 1px ${g.color}40` }}
        >
          GRAP {g.stage}
        </span>
      </div>
      <div className="mt-2 text-2xs text-mute">
        {ward.pop_2026.toLocaleString('en-IN')} residents · {ward.udise_schools} schools · {ward.civic_hospitals} civic hospitals
      </div>
      <Button variant="outline" className="mt-3 w-full" onClick={() => setView('wards')}>
        Open ward actions <ArrowRight size={14} />
      </Button>
    </div>
  )
}
