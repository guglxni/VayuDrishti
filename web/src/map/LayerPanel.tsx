import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { LAYER_LABELS, useTwin, type LayerKey } from '../state/store'
import { AqiLegend, Button, cx, Toggle } from '../ui/primitives'
import type { Cam } from './MapView'

const ORDER: LayerKey[] = ['surface', 'wind', 'airphynet', 'network', 'wards', 'buildings', 'sources', 'facilities']
const CAMERA_PRESETS: { id: Cam; label: string; title: string }[] = [
  { id: 'sea', label: 'From sea', title: 'Look east across the city from the Arabian Sea' },
  { id: 'north', label: 'North up', title: 'North-up city overview' },
  { id: 'deonar', label: 'Shivaji / Deonar', title: 'Shivaji Nagar and the Deonar emission plume' },
  { id: 'malad', label: 'Malad West', title: 'Malad West forecast station' },
  { id: 'borivali', label: 'Borivali East', title: 'Borivali East forecast station' },
  { id: 'byculla', label: 'Byculla', title: 'Byculla forecast station' },
]

export function LayerPanel({ onCamera }: { onCamera: (cam: Cam) => void }) {
  const layers = useTwin((s) => s.layers)
  const toggle = useTwin((s) => s.toggleLayer)
  const [open, setOpen] = useState(true)

  return (
    <aside className="panel absolute right-5 top-20 z-10 w-[268px]" aria-label="Map layers">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span>
          <span className="block text-[13px] font-medium">PM2.5, µg/m³</span>
          <span className="block text-2xs text-faint">CPCB air quality index bands</span>
        </span>
        <ChevronDown size={16} className={cx('text-mute transition-transform', open ? 'rotate-180' : '')} />
      </button>
      <div className="px-4 pb-3">
        <AqiLegend />
      </div>
      {open && (
        <>
          <div className="border-t border-line px-2 py-2">
            {ORDER.map((k) => (
              <Toggle
                key={k}
                checked={layers[k]}
                onChange={() => toggle(k)}
                label={LAYER_LABELS[k].label}
                hint={LAYER_LABELS[k].hint}
              />
            ))}
          </div>
          <div className="border-t border-line px-3 py-3">
            <div className="eyebrow mb-2 px-1">Camera presets</div>
            <div className="grid grid-cols-2 gap-1">
              {CAMERA_PRESETS.map((cam) => (
                <Button key={cam.id} className="justify-start px-2 text-[11px]" onClick={() => onCamera(cam.id)} title={cam.title}>
                  {cam.label}
                </Button>
              ))}
            </div>
          </div>
        </>
      )}
    </aside>
  )
}
