import { useState, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { useData, useTwin } from '../../state/store'
import { stationPM } from '../../lib/series'
import { BLOCK_LABEL, TARGET_IDS, TARGET_META, istParts, targetStations, stationName } from '../../lib/stations'
import { AqiChip, PmValue, Segmented } from '../../ui/primitives'
import { FanChart } from './FanChart'
import { AccuracySection } from './AccuracySection'
import { WeightsSection } from './WeightsSection'
import { MeanMedianSection } from './MeanMedianSection'
import { HampelSection } from './HampelSection'
import { AnalogSection } from './AnalogSection'
import { usePrefersReducedMotion } from './shared'

type RangeKey = 'scored' | 'cursor' | 'month' | 'validation'

const RANGE_OPTIONS: { value: RangeKey; label: string; title: string }[] = [
  { value: 'scored', label: 'Scored window', title: '11–12 Nov, the leaderboard-scored 48 hours, with a 12-hour margin' },
  { value: 'cursor', label: 'Around cursor', title: '24 hours before to 48 hours after the time cursor' },
  { value: 'month', label: 'Forecast month', title: '22 Oct – 20 Nov' },
  { value: 'validation', label: 'Validation', title: '12–21 Oct: out-of-fold forecasts against observations' },
]

const FALLBACK_STATION = 3409511

export function ForecastView() {
  const data = useData()
  const fc = data.forecasts
  const hour = useTwin((s) => s.hour)
  const setHour = useTwin((s) => s.setHour)
  const stationId = useTwin((s) => s.stationId)
  const setStation = useTwin((s) => s.setStation)
  const [rangeKey, setRangeKey] = useState<RangeKey>('scored')
  const reduced = usePrefersReducedMotion()

  const sid = (TARGET_IDS as readonly number[]).includes(stationId) && fc.stations[String(stationId)] ? stationId : FALLBACK_STATION
  const stationLabel = TARGET_META[sid]?.name ?? String(sid)
  const last = fc.hours.length - 1
  const priv = fc.windows.find((w) => w.id === 'private')
  const val = fc.windows.find((w) => w.id === 'validation')

  const range: [number, number] = (() => {
    switch (rangeKey) {
      case 'scored':
        return priv ? [priv.start - 12, priv.end + 12] : [708, 779]
      case 'cursor': {
        const s = Math.max(0, Math.min(last - 72, hour - 24))
        return [s, Math.min(last, s + 72)]
      }
      case 'month':
        return [val ? val.end + 1 : 240, last]
      case 'validation':
        return val ? [val.start, val.end] : [0, 239]
    }
  })()

  const pm = stationPM(data, sid, hour)
  const obs = fc.stations[String(sid)]?.obs[hour] ?? null
  const p = istParts(fc.hours[hour])
  const block = fc.blocks[hour]
  const cursorOutside = hour < range[0] || hour > range[1]

  const stations = targetStations(data)

  return (
    <div className="h-full overflow-y-auto pb-16 pt-[88px]">
      <div className="mx-auto max-w-[1280px] px-6">
        {/* Header */}
        <header className="flex flex-wrap items-end justify-between gap-x-8 gap-y-5">
          <div className="min-w-0 max-w-[620px]">
            <div className="eyebrow mb-1.5">Forecast · {TARGET_META[sid]?.area}</div>
            <h1 className="text-[28px] font-semibold leading-tight tracking-tight text-paper">Station forecast</h1>
            <p className="mt-1.5 text-[14px] leading-relaxed text-mute">
              Hour-by-hour PM2.5 at the four scored monitors, issued at midnight on 22 Oct with no later data. The shaded fan
              shows how sure the model is; the line is the value we submit.
            </p>
          </div>
          <div className="flex items-end gap-6">
            <div className="text-right">
              <div className="text-2xs text-faint">
                {stationLabel} · {p.day} {p.date}, {p.time}
              </div>
              <div className="mt-1 flex items-center justify-end gap-3">
                <AqiChip value={pm} />
                <PmValue value={pm} size="lg" />
              </div>
              <div className="mt-1 text-2xs text-mute">
                {BLOCK_LABEL[block]}
                {obs != null && (
                  <>
                    <span className="text-faint"> · </span>observed <span className="font-mono tnum text-paper">{obs.toFixed(1)}</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </header>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-y border-line py-3">
          <Segmented
            value={sid}
            onChange={setStation}
            options={stations.map((s) => ({ value: s.id, label: stationName(s), title: TARGET_META[s.id]?.area }))}
          />
          <Segmented<RangeKey> value={rangeKey} onChange={setRangeKey} options={RANGE_OPTIONS} size="sm" />
        </div>

        {/* Hero */}
        <Reveal reduced={reduced} className="mt-6">
          <FanChart fc={fc} sid={sid} range={range} hour={hour} setHour={setHour} />
          {cursorOutside && (
            <p className="mt-2 text-2xs text-faint">
              The time cursor ({p.date}, {p.time}) is outside this range. Click the chart to move it, or pick “Around cursor”.
            </p>
          )}
        </Reveal>

        <div className="mt-16 space-y-16">
          <Reveal reduced={reduced}>
            <AccuracySection fc={fc} />
          </Reveal>

          <div className="grid gap-x-8 gap-y-16 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Reveal reduced={reduced}>
              <WeightsSection fc={fc} stationId={sid} />
            </Reveal>
            <Reveal reduced={reduced}>
              <MeanMedianSection fc={fc} sid={sid} hour={hour} stationLabel={stationLabel} />
            </Reveal>
          </div>

          <Reveal reduced={reduced}>
            <HampelSection fc={fc} sid={sid} stationLabel={stationLabel} />
          </Reveal>

          <Reveal reduced={reduced}>
            <AnalogSection fc={fc} sid={sid} hour={hour} setHour={setHour} stationLabel={stationLabel} />
          </Reveal>

          <footer className="border-t border-line pt-5 text-2xs leading-relaxed text-faint">
            Times are IST hour-start. Forecast issued {fc.meta.cutoff_t0.slice(0, 10)} 00:00 IST using only readings whose hour
            had ended by then. Uncertainty bands: {fc.meta.provenance.bands}
          </footer>
        </div>
      </div>
    </div>
  )
}

function Reveal({ children, reduced, className }: { children: ReactNode; reduced: boolean; className?: string }) {
  if (reduced) return <div className={className}>{children}</div>
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.45, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </motion.div>
  )
}
