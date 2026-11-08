import { useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import type { Forecasts, Num } from '../../data/types'
import { NAAQS_24H, pmColor } from '../../lib/aqi'
import { fmt } from '../../lib/series'
import { istParts } from '../../lib/stations'
import { Button, cx } from '../../ui/primitives'
import { C, EmptyNote, SectionHead, useWidth } from './shared'

const H = 260
const M = { top: 14, right: 14, bottom: 28, left: 40 }

type P = { h: number; v: number }

function toPts(arr: Num[]): (P | null)[] {
  return arr.map((v, h) => (v == null ? null : { h, v }))
}

export function AnalogSection({
  fc,
  sid,
  hour,
  setHour,
  stationLabel,
}: {
  fc: Forecasts
  sid: number
  hour: number
  setHour: (h: number) => void
  stationLabel: string
}) {
  const [ref, width] = useWidth<HTMLDivElement>(560)
  const [hoverA, setHoverA] = useState<string | null>(null)
  const iso = fc.hours[hour]
  const date = iso.slice(0, 10)
  const entry = fc.analogs[String(sid)]?.[date]
  const p = istParts(iso)

  const head = (
    <SectionHead
      eyebrow="Analog ensemble"
      title="Days with similar weather"
      subtitle={`For ${p.day} ${p.date} at ${stationLabel}: past days whose wind, mixing height and humidity looked most alike, and how PM2.5 played out on them.`}
    />
  )

  if (!entry) {
    const firstDate = Object.keys(fc.analogs[String(sid)] ?? {}).sort()[0]
    const firstIdx = firstDate ? fc.hours.findIndex((h) => h.startsWith(firstDate)) : -1
    const priv = fc.windows.find((w) => w.id === 'private')
    return (
      <section>
        {head}
        <EmptyNote>
          Analog days are matched for forecast dates from 22 Oct onward. {p.date} is in the validation period, where
          observations are shown instead.
          <div className="mt-4 flex justify-center gap-2">
            {firstIdx >= 0 && (
              <Button variant="outline" onClick={() => setHour(firstIdx + 8)}>
                Go to 22 Oct
              </Button>
            )}
            {priv && (
              <Button variant="outline" onClick={() => setHour(priv.start + 8)}>
                Go to the scored window
              </Button>
            )}
          </div>
        </EmptyNote>
      </section>
    )
  }

  const dayStart = fc.hours.findIndex((h) => h.startsWith(`${date}T00`))
  const ens: Num[] = dayStart >= 0 ? (fc.stations[String(sid)].models.ensemble ?? []).slice(dayStart, dayStart + 24) : []
  const hod = Number(iso.slice(11, 13))

  const analogs = entry.analogs.slice(0, 3)
  const all = [...analogs.flatMap((a) => a.curve), ...entry.anen_curve, ...ens].filter((v): v is number => v != null)
  const iw = Math.max(200, width - M.left - M.right)
  const ih = H - M.top - M.bottom
  const x = scaleLinear().domain([0, 23]).range([0, iw])
  const y = scaleLinear()
    .domain([0, Math.max(NAAQS_24H + 10, ...all) * 1.08])
    .range([ih, 0])
    .nice()
  const ln = line<P | null>()
    .defined((d) => d != null)
    .x((d) => x(d!.h))
    .y((d) => y(d!.v))
  const ensNow = ens[hod] ?? null

  const met = entry.target_met

  return (
    <section>
      {head}
      <div className="panel-solid grid gap-6 p-5 lg:grid-cols-[300px_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="eyebrow mb-2">Forecast weather, daily mean</div>
          <div className="grid grid-cols-4 gap-3 border-b border-line pb-4">
            <MetFig label="Wind" value={fmt(met.wind_ms, 1)} unit="m/s" />
            <MetFig label="Mixing ht." value={fmt(met.pblh_m, 0)} unit="m" />
            <MetFig label="Humidity" value={fmt(met.rh, 0)} unit="%" />
            <MetFig label="Temp." value={fmt(met.temp_c, 1)} unit="°C" />
          </div>
          <div className="eyebrow mb-1 mt-4">Closest past days</div>
          <ul>
            {analogs.map((a) => {
              const ap = istParts(`${a.date}T00:00`)
              const on = hoverA === a.date
              return (
                <li
                  key={a.date}
                  onPointerEnter={() => setHoverA(a.date)}
                  onPointerLeave={() => setHoverA(null)}
                  className={cx('-mx-2 rounded-lg px-2 py-2 transition-colors', on && 'bg-paper/[0.05]')}
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[13px] text-paper">
                      {ap.day} {ap.date}
                    </span>
                    <span className="font-mono text-[11px] tnum text-mute">
                      mean <span className="text-paper">{fmt(a.pm25_mean)}</span> µg/m³
                    </span>
                  </div>
                  <div className="mt-1.5 flex items-center gap-2">
                    <div className="h-1 flex-1 overflow-hidden rounded-full bg-paper/10">
                      <div className="h-full rounded-full bg-paper/70" style={{ width: `${a.similarity * 100}%` }} />
                    </div>
                    <span className="w-[86px] text-right font-mono text-[11px] tnum text-mute">
                      {Math.round(a.similarity * 100)}% similar
                    </span>
                  </div>
                  <div className="mt-1 font-mono text-[10px] tnum text-faint">
                    {fmt(a.wind_ms, 1)} m/s · {fmt(a.pblh_m, 0)} m · {fmt(a.rh, 0)}% RH
                  </div>
                </li>
              )
            })}
          </ul>
        </div>

        <div ref={ref} className="min-w-0">
          <svg width={width} height={H} role="img" aria-label="Hourly PM2.5 on analog days compared with the forecast">
            <g transform={`translate(${M.left},${M.top})`}>
              {y.ticks(5).map((v) => (
                <g key={v}>
                  <line x1={0} x2={iw} y1={y(v)} y2={y(v)} stroke={v === 0 ? C.gridStrong : C.grid} />
                  <text x={-10} y={y(v)} dy="0.32em" textAnchor="end" className="font-mono" fontSize={11} fill={C.faint}>
                    {v}
                  </text>
                </g>
              ))}
              {[0, 6, 12, 18, 23].map((hh) => (
                <text key={hh} x={x(hh)} y={ih + 18} textAnchor="middle" className="font-mono" fontSize={11} fill={C.faint}>
                  {String(hh).padStart(2, '0')}:00
                </text>
              ))}
              <line x1={0} x2={iw} y1={y(NAAQS_24H)} y2={y(NAAQS_24H)} stroke="rgba(236,231,220,0.3)" strokeDasharray="5 4" />
              <text x={iw - 4} y={y(NAAQS_24H) - 6} textAnchor="end" fontSize={11} fill={C.faint}>
                60
              </text>

              {analogs.map((a) => {
                const on = hoverA === a.date
                return (
                  <path
                    key={a.date}
                    d={ln(toPts(a.curve)) ?? ''}
                    fill="none"
                    stroke={on ? C.paper : C.mute}
                    strokeOpacity={on ? 0.9 : hoverA ? 0.2 : 0.4}
                    strokeWidth={on ? 1.5 : 1}
                  />
                )
              })}
              <path d={ln(toPts(entry.anen_curve)) ?? ''} fill="none" stroke={C.paper} strokeOpacity={0.8} strokeWidth={1.5} strokeDasharray="5 3" />
              {ens.length > 0 && (
                <>
                  <path d={ln(toPts(ens)) ?? ''} fill="none" stroke={C.ink2} strokeWidth={4.5} />
                  <path d={ln(toPts(ens)) ?? ''} fill="none" stroke={C.paper} strokeWidth={2} strokeLinejoin="round" />
                </>
              )}
              <line x1={x(hod)} x2={x(hod)} y1={0} y2={ih} stroke={C.paper} strokeOpacity={0.4} />
              {ensNow != null && <circle cx={x(hod)} cy={y(ensNow)} r={4.5} fill={pmColor(ensNow)} stroke={C.ink2} strokeWidth={2} />}
            </g>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 pl-10 text-2xs text-faint">
            <span className="inline-flex items-center gap-1.5">
              <svg width="16" height="6" aria-hidden>
                <line x1="0" x2="16" y1="3" y2="3" stroke={C.paper} strokeWidth={2} />
              </svg>
              Ensemble forecast
            </span>
            <span className="inline-flex items-center gap-1.5">
              <svg width="16" height="6" aria-hidden>
                <line x1="0" x2="16" y1="3" y2="3" stroke={C.paper} strokeDasharray="5 3" strokeWidth={1.5} />
              </svg>
              Analog-ensemble estimate
            </span>
            <span className="inline-flex items-center gap-1.5">
              <svg width="16" height="6" aria-hidden>
                <line x1="0" x2="16" y1="3" y2="3" stroke={C.mute} strokeOpacity={0.6} />
              </svg>
              Individual analog days (hover a day)
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}

function MetFig({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-2xs text-faint">{label}</div>
      <div className="flex items-baseline gap-0.5">
        <span className="font-cond text-[17px] font-medium tnum text-paper">{value}</span>
        <span className="text-[10px] text-mute">{unit}</span>
      </div>
    </div>
  )
}
