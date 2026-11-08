import { ArrowRight, Navigation } from 'lucide-react'
import { useData, useTwin } from '../state/store'
import { pmColor } from '../lib/aqi'
import { REGIME_LABEL, cityMet } from '../lib/physics'
import { fmt, sampleAt, stationPM } from '../lib/series'
import { istParts, stationName, TARGET_META, targetStations } from '../lib/stations'
import { AqiChip, Button, cx, PmValue } from '../ui/primitives'

const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

export function compass(deg: number) {
  return COMPASS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16]
}

export function StationCard() {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const stationId = useTwin((s) => s.stationId)
  const setStation = useTwin((s) => s.setStation)
  const setView = useTwin((s) => s.setView)

  const st = data.stations.stations.find((s) => s.id === stationId)
  const fc = data.forecasts.stations[String(stationId)]
  if (!st || !fc) return null

  const iso = data.forecasts.hours[hour]
  const when = istParts(iso)
  const block = data.forecasts.blocks[hour]
  const pm = stationPM(data, stationId, hour)
  const observed = block === 'validation' ? fc.obs[hour] : null
  const dec = fc.deciles[hour]
  const lo = dec?.[0]
  const hi = dec?.[8]

  let peak: { v: number; i: number } | null = null
  for (let i = hour + 1; i <= Math.min(hour + 24, fc.models.ensemble.length - 1); i++) {
    const v = fc.models.ensemble[i]
    if (v != null && (!peak || v > peak.v)) peak = { v, i }
  }

  const wp = data.wind.points[String(stationId)]
  const u = sampleAt(wp?.u, hour) ?? 0
  const v = sampleAt(wp?.v, hour) ?? 0
  const speed = Math.hypot(u, v)
  const fromDeg = ((Math.atan2(-u, -v) * 180) / Math.PI + 360) % 360
  const pblh = sampleAt(wp?.pblh, hour)
  const met = cityMet(data, hour)

  return (
    <aside className="panel absolute left-5 top-20 z-10 w-[340px] overflow-hidden" aria-label="Selected station">
      <div className="px-5 pb-4 pt-4">
        <div className="eyebrow">
          Forecast station · {TARGET_META[stationId]?.operator ?? st.operator} {stationId}
        </div>
        <h2 className="mt-1 text-[19px] font-semibold tracking-tight">{stationName(st)}</h2>
        <p className="text-xs text-mute">{TARGET_META[stationId]?.area ?? st.regime}</p>

        <div className="mt-4 flex items-end justify-between gap-3">
          <PmValue value={pm} size="xl" />
          <AqiChip value={pm} className="mb-2" />
        </div>
        <div className="mt-1.5 text-xs text-mute">
          {block === 'validation' ? 'Model hindcast' : 'Forecast'} for {when.day} {when.date}, {when.time} IST
          {observed != null && (
            <>
              {' · '}
              <span className="text-paper">observed {fmt(observed)}</span>
            </>
          )}
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3 border-t border-line pt-3">
          <div>
            <div className="text-2xs text-faint">Likely range (80%)</div>
            <div className="font-cond text-[17px] font-medium tnum">
              {lo != null && hi != null ? `${fmt(lo, 0)}–${fmt(hi, 0)}` : '—'}
              <span className="ml-1 font-sans text-2xs text-mute">µg/m³</span>
            </div>
          </div>
          <div>
            <div className="text-2xs text-faint">Peak in next 24 h</div>
            <div className="font-cond text-[17px] font-medium tnum">
              <span style={{ color: pmColor(peak?.v) }}>{fmt(peak?.v, 0)}</span>
              <span className="ml-1 font-sans text-2xs text-mute">at {istParts(data.forecasts.hours[peak?.i ?? hour]).time}</span>
            </div>
          </div>
          <div>
            <div className="text-2xs text-faint">Wind</div>
            <div className="flex items-center gap-1.5 font-cond text-[17px] font-medium tnum text-wind">
              <Navigation size={13} style={{ transform: `rotate(${fromDeg + 180}deg)` }} aria-hidden />
              {fmt(speed)} <span className="font-sans text-2xs text-mute">m/s from {compass(fromDeg)}</span>
            </div>
          </div>
          <div>
            <div className="text-2xs text-faint">Mixing height</div>
            <div className="font-cond text-[17px] font-medium tnum">
              {fmt(pblh, 0)}
              <span className="ml-1 font-sans text-2xs text-mute">m</span>
            </div>
          </div>
        </div>
        <p className="mt-3 text-xs text-mute">{REGIME_LABEL[met.regime] ?? met.regime}</p>
      </div>

      <div className="border-t border-line px-3 py-2">
        <div className="px-2 pb-1 text-2xs text-faint">All forecast stations</div>
        <ul>
          {targetStations(data).map((s) => {
            const val = stationPM(data, s.id, hour)
            const active = s.id === stationId
            return (
              <li key={s.id}>
                <button
                  onClick={() => setStation(s.id)}
                  className={cx(
                    'flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors',
                    active ? 'bg-paper/8' : 'hover:bg-paper/5',
                  )}
                  aria-pressed={active}
                >
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: pmColor(val) }} />
                  <span className={cx('flex-1 text-[13px]', active ? 'text-paper' : 'text-mute')}>{stationName(s)}</span>
                  <span className="font-cond text-[15px] font-medium tnum" style={{ color: pmColor(val) }}>
                    {fmt(val, 0)}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      </div>

      <div className="flex gap-1 border-t border-line px-3 py-2.5">
        <Button variant="outline" className="flex-1" onClick={() => setView('column')}>
          See the air column <ArrowRight size={14} />
        </Button>
        <Button variant="ghost" onClick={() => setView('forecast')}>
          Forecast
        </Button>
      </div>
    </aside>
  )
}
