import { useState } from 'react'
import { ChevronDown, Compass, Waves } from 'lucide-react'
import { LAYER_LABELS, useTwin, type LayerKey } from '../state/store'
import { AqiLegend, Button, cx, Toggle } from '../ui/primitives'

const ORDER: LayerKey[] = ['surface', 'wind', 'network', 'wards', 'buildings', 'sources', 'facilities']

export function LayerPanel({ onCamera }: { onCamera: (cam: 'sea' | 'north') => void }) {
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
          <div className="flex gap-1 border-t border-line px-2 py-2">
            <Button className="flex-1" onClick={() => onCamera('sea')} title="Look east across the city from the Arabian Sea">
              <Waves size={14} /> From the sea
            </Button>
            <Button className="flex-1" onClick={() => onCamera('north')} title="North-up overview">
              <Compass size={14} /> North up
            </Button>
          </div>
        </>
      )}
    </aside>
  )
}
