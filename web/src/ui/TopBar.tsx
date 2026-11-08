import { Info, Pause, Play } from 'lucide-react'
import { useTwin, type View } from '../state/store'
import { istParts } from '../lib/stations'
import { Button, Segmented } from './primitives'

const VIEWS: { value: View; label: string; title: string }[] = [
  { value: 'map', label: 'City map', title: 'PM2.5, wind and stations on a 3D map of Mumbai' },
  { value: 'column', label: 'Air column', title: 'How the boundary layer traps pollution above a station' },
  { value: 'forecast', label: 'Forecast', title: 'Station forecasts, uncertainty and model accuracy' },
  { value: 'wards', label: 'Ward actions', title: 'Ward risk ranking and GRAP response orders' },
]

function Mark() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7 shrink-0" aria-hidden>
      <circle cx="16" cy="16" r="15" fill="none" stroke="currentColor" strokeOpacity="0.25" />
      <path d="M5 13c4-3 7 2 11 0s7-3 11 0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M5 18c4-3 7 2 11 0s7-3 11 0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeOpacity="0.6" />
      <path d="M8 23c3-2 5 1.5 8 0s5-2 8 0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeOpacity="0.3" />
    </svg>
  )
}

/** Compact clock shown when the full timeline is hidden. */
function TimeChip() {
  const hour = useTwin((s) => s.hour)
  const iso = useTwin((s) => s.data?.forecasts.hours[s.hour])
  const playing = useTwin((s) => s.playing)
  const setPlaying = useTwin((s) => s.setPlaying)
  const setHour = useTwin((s) => s.setHour)
  const p = istParts(iso)
  return (
    <div className="flex items-center gap-1 rounded-[10px] border border-line bg-ink/60 py-0.5 pl-0.5 pr-3">
      <button
        aria-label={playing ? 'Pause' : 'Play'}
        onClick={() => setPlaying((v) => !v)}
        className="flex h-7 w-7 items-center justify-center rounded-[8px] text-paper hover:bg-paper/10"
      >
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <button className="px-1 text-mute hover:text-paper" aria-label="Previous hour" onClick={() => setHour(hour - 1)}>
        ‹
      </button>
      <span className="font-mono text-xs tnum text-paper">
        {p.day} {p.date} · {p.time}
      </span>
      <button className="px-1 text-mute hover:text-paper" aria-label="Next hour" onClick={() => setHour(hour + 1)}>
        ›
      </button>
    </div>
  )
}

export function TopBar() {
  const view = useTwin((s) => s.view)
  const setView = useTwin((s) => s.setView)
  const setAboutOpen = useTwin((s) => s.setAboutOpen)
  const solidBg = view === 'forecast' || view === 'wards'

  return (
    <header
      className={
        'pointer-events-none absolute inset-x-0 top-0 z-30 flex h-16 items-center gap-4 px-5 ' +
        (solidBg
          ? 'border-b border-line bg-ink/92 backdrop-blur'
          : 'bg-gradient-to-b from-ink/90 via-ink/55 to-transparent')
      }
    >
      <div className="pointer-events-auto flex min-w-0 items-center gap-3 text-paper">
        <Mark />
        <div className="min-w-0 leading-tight">
          <div className="flex items-baseline gap-2">
            <span className="font-deva text-[19px] leading-none">वायुदृष्टि</span>
            <span className="text-[15px] font-semibold tracking-tight">VayuDrishti</span>
            <span className="hidden text-[15px] font-light text-mute sm:inline">Mumbai</span>
          </div>
          <div className="hidden truncate text-2xs text-mute lg:block">
            Hourly PM2.5 forecast for Mumbai's air, 22 Oct – 20 Nov 2026
          </div>
        </div>
      </div>

      <nav className="pointer-events-auto mx-auto">
        <Segmented options={VIEWS} value={view} onChange={setView} />
      </nav>

      <div className="pointer-events-auto flex items-center gap-2">
        {(view === 'forecast' || view === 'wards') && <TimeChip />}
        <Button variant="ghost" onClick={() => setAboutOpen(true)}>
          <Info size={15} />
          <span className="hidden md:inline">About the model</span>
        </Button>
      </div>
    </header>
  )
}
