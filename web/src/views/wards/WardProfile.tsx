import { Hospital, School } from 'lucide-react'
import type { Facility, Ward } from '../../data/types'
import { NAAQS_24H, pmColor, WHO_24H } from '../../lib/aqi'
import { fmt, sampleAt } from '../../lib/series'
import { useData, useTwin } from '../../state/store'
import { AqiChip, Eyebrow, PmValue, Stat } from '../../ui/primitives'
import { Locator } from './Locator'
import { GrapChip } from './parts'
import { WardChart } from './WardChart'
import { fmtInt, fmtMultiple, peakAhead, whenLabel } from './util'

export function WardProfile({ ward }: { ward: Ward }) {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const setHour = useTwin((s) => s.setHour)
  const setWard = useTwin((s) => s.setWard)
  const hours = data.forecasts.hours

  const pm = sampleAt(ward.series, hour)
  const peak = peakAhead(ward.series, hour, 24)
  const named = data.wards.facilities.filter((f) => f.ward === ward.code && !f.indicative)
  const pins = data.wards.facilities.filter((f) => f.ward === ward.code && f.indicative).length
  const meta = data.wards.meta as Record<string, string | undefined>

  return (
    <div className="space-y-4 px-4 pb-6 pt-1 min-[1100px]:px-2">
      {/* Header */}
      <section className="panel-solid p-5">
        <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-4">
          <div className="min-w-0">
            <Eyebrow>
              Ward {ward.code} · {ward.zone}
            </Eyebrow>
            <h1 className="mt-1 text-[22px] font-semibold leading-tight tracking-tight text-paper">{ward.name}</h1>
            <div className="mt-1 font-mono text-2xs tnum text-faint">{whenLabel(hours[hour])} · ward estimate</div>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <PmValue value={pm} size="lg" />
            <div className="flex items-center gap-1.5">
              <AqiChip value={pm} />
              <GrapChip value={pm} />
            </div>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-4 sm:grid-cols-4">
          <div className="min-w-0">
            <div className="text-2xs text-faint">Peak in next 24 h</div>
            <div className="mt-0.5 flex items-baseline gap-1">
              <span className="font-cond text-lg font-semibold tnum" style={{ color: pmColor(peak?.value) }}>
                {fmt(peak?.value, 0)}
              </span>
              <span className="text-2xs text-mute">µg/m³</span>
            </div>
            <div className="text-2xs text-mute">{peak ? whenLabel(hours[peak.index]) : 'No estimate ahead'}</div>
          </div>
          <Stat label="Peak stage" value={<GrapChip value={peak?.value ?? null} className="align-middle" />} hint="GRAP threshold at the 24-h peak" />
          <Stat label="× NAAQS now" value={fmtMultiple(pm, NAAQS_24H, 2)} hint={`India 24-h standard, ${NAAQS_24H} µg/m³`} />
          <Stat label="× WHO now" value={fmtMultiple(pm, WHO_24H, 1)} hint={`WHO 2021 guideline, ${WHO_24H} µg/m³`} />
        </div>
      </section>

      {/* Chart */}
      <section className="panel-solid px-4 pb-3 pt-4">
        <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2 px-1">
          <h2 className="text-[13px] font-medium text-paper">PM2.5 · last 24 h and next 72 h</h2>
          <span className="text-2xs text-faint">Click the chart or use ← → to move the hour</span>
        </div>
        <WardChart series={ward.series} hours={hours} hour={hour} windows={data.forecasts.windows} onPick={(i) => setHour(i)} />
      </section>

      {/* Facts + locator */}
      <section className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_200px]">
        <div className="panel-solid p-5">
          <h2 className="text-[13px] font-medium text-paper">Who and what is exposed</h2>
          <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-5">
            <Stat label="Residents" value={fmtInt(ward.pop_2026)} hint="WorldPop 2026 estimate" />
            <Stat
              label="Vulnerable people"
              value={fmtInt(ward.vulnerable_pop)}
              hint={`Children under 5 ${fmt(ward.children_pct, 1)}% · over 60 ${fmt(ward.elderly_pct, 1)}%`}
            />
            <Stat label="Informal settlement share" value={fmt(ward.slum_pct, 0)} unit="%" hint="Of ward residents" />
            <Stat label="Active construction sites" value={fmtInt(ward.maharera_sites)} hint="MahaRERA registered" />
            <Stat label="Schools" value={fmtInt(ward.udise_schools)} hint="UDISE+ register" />
            <Stat label="Civic hospitals" value={fmtInt(ward.civic_hospitals)} hint="BMC run" />
            <Stat
              className="col-span-2"
              label="Forecast days above NAAQS"
              value={
                <>
                  {ward.naaqs_exceed_days}
                  <span className="text-[13px] font-normal text-mute"> of {ward.forecast_days} days</span>
                </>
              }
              hint={`Daily mean above ${NAAQS_24H} µg/m³ · above the WHO guideline on ${ward.who_exceed_days} of ${ward.forecast_days} days`}
            />
          </div>
        </div>
        <div className="panel-solid flex flex-col p-4">
          <h2 className="text-[13px] font-medium text-paper">Location</h2>
          <div className="mt-3 flex-1">
            <Locator wards={data.wards.wards} outline={data.wards.outline} hour={hour} selected={ward.code} onSelect={setWard} />
          </div>
          <p className="mt-3 text-[10px] leading-snug text-faint" title={meta.geometry}>
            Approximate ward cells, not official boundaries. Fill shows PM2.5 now.
          </p>
        </div>
      </section>

      {/* Facilities */}
      <section className="panel-solid p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-[13px] font-medium text-paper">Named schools and hospitals</h2>
          <span className="text-2xs text-faint">
            {named.length} named · {ward.udise_schools} schools and {ward.civic_hospitals} civic hospitals in the ward
          </span>
        </div>
        {named.length ? (
          <ul className="mt-3 divide-y divide-line">
            {named.map((f) => (
              <FacilityRow key={f.id} f={f} />
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-[13px] leading-relaxed text-mute">
            No named facilities are curated for this ward yet. The ward totals above still drive the response order
            {pins ? `; the city map shows ${pins} indicative school pins here (one pin stands for about 12 schools).` : '.'}
          </p>
        )}
      </section>
    </div>
  )
}

function FacilityRow({ f }: { f: Facility }) {
  const Icon = f.type === 'hospital' ? Hospital : School
  return (
    <li className="flex gap-3 py-3 first:pt-1 last:pb-0">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-line text-mute">
        <Icon size={14} aria-label={f.type === 'hospital' ? 'Hospital' : 'School'} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-[13px] font-medium text-paper">{f.name}</span>
          {f.tier && <span className="text-2xs text-mute">{f.tier}</span>}
        </div>
        {f.protocol && <p className="mt-0.5 text-[12.5px] leading-snug text-mute">{f.protocol}</p>}
        {f.daily_population != null && (
          <p className="mt-0.5 text-2xs text-faint">About {f.daily_population.toLocaleString('en-IN')} people on site daily</p>
        )}
      </div>
    </li>
  )
}
