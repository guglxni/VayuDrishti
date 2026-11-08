import { useMemo, useState } from 'react'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import type { Forecasts, HampelEvent } from '../../data/types'
import { Segmented, cx } from '../../ui/primitives'
import { C, EmptyNote, SectionHead, dayLabel, longStamp, useWidth } from './shared'

const CAP = 450
const H = 250
const M = { top: 16, right: 14, bottom: 28, left: 52 }

type StageFilter = 'all' | 1 | 2

const numFmt = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 1 })
const intFmt = new Intl.NumberFormat('en-IN')

function logTick(v: number): string {
  if (v >= 1000) return `${v / 1000}k`
  return String(v)
}

export function HampelSection({ fc, sid, stationLabel }: { fc: Forecasts; sid: number; stationLabel: string }) {
  const h = fc.hampel[String(sid)]
  const [filter, setFilter] = useState<StageFilter>('all')
  const [selKey, setSelKey] = useState<string | null>(null)

  const events = useMemo(() => (h ? h.events.filter((e) => filter === 'all' || e.stage === filter) : []), [h, filter])
  const selected = events.find((e) => e.hour_start === selKey) ?? events[0] ?? null

  const head = (
    <SectionHead
      eyebrow="Data cleaning"
      title="Sensor glitches removed before training"
      subtitle={`${stationLabel}'s history contains impossible readings. Models learn from a cleaned series; accuracy is still scored on the raw readings.`}
    />
  )

  if (!h) {
    return (
      <section>
        {head}
        <EmptyNote>No cleaning log is available for this station.</EmptyNote>
      </section>
    )
  }

  const n1 = h.events.filter((e) => e.stage === 1).length
  const n2 = h.events.filter((e) => e.stage === 2).length

  return (
    <section>
      {head}
      <div className="panel-solid p-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 border-b border-line pb-5 md:grid-cols-4">
          <Figure label="Hourly readings checked" value={intFmt.format(h.n_rows)} />
          <Figure
            label="Hard-cap corrections"
            value={intFmt.format(h.n_stage1)}
            hint={`flagged by CPCB or above ${CAP} µg/m³`}
          />
          <Figure label="Spike corrections" value={intFmt.format(h.n_stage2)} hint="7-hour Hampel filter, 4.5 × robust spread" />
          <Figure label="Largest raw reading" value={numFmt.format(h.max_raw)} unit="µg/m³" hint="physically impossible" />
        </div>

        <div className="mt-5 grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
          <div className="min-w-0">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="eyebrow">Largest corrections</span>
              {n1 > 0 && n2 > 0 && (
                <Segmented<StageFilter>
                  size="sm"
                  value={filter}
                  onChange={setFilter}
                  options={[
                    { value: 'all', label: 'All' },
                    { value: 1, label: 'Cap' },
                    { value: 2, label: 'Spike' },
                  ]}
                />
              )}
            </div>
            <ul className="max-h-[300px] overflow-y-auto pr-1" role="listbox" aria-label="Corrected readings">
              {events.map((e) => {
                const on = selected?.hour_start === e.hour_start
                return (
                  <li key={e.hour_start}>
                    <button
                      role="option"
                      aria-selected={on}
                      onClick={() => setSelKey(e.hour_start)}
                      className={cx(
                        'grid w-full grid-cols-[1fr_auto] items-baseline gap-x-3 rounded-lg px-2.5 py-1.5 text-left transition-colors',
                        on ? 'bg-paper/[0.07]' : 'hover:bg-paper/[0.04]',
                      )}
                    >
                      <span className={cx('font-mono text-[11px] tnum', on ? 'text-paper' : 'text-mute')}>
                        {longStamp(e.hour_start)}
                      </span>
                      <span className="font-mono text-[11px] tnum">
                        <span className={on ? 'text-paper' : 'text-mute'}>{numFmt.format(e.raw)}</span>
                        <span className="text-faint"> → </span>
                        <span className="text-paper">{numFmt.format(e.clean)}</span>
                      </span>
                      <span className="col-span-2 text-[10px] text-faint">
                        {e.stage === 1 ? (e.flag_high_1000 ? 'Hard cap · flagged by CPCB' : `Hard cap · above ${CAP}`) : 'Spike · Hampel filter'}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
          <div className="min-w-0">{selected ? <EventChart e={selected} /> : <EmptyNote>No events in this filter.</EmptyNote>}</div>
        </div>
      </div>
    </section>
  )
}

function Figure({ label, value, unit, hint }: { label: string; value: string; unit?: string; hint?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-2xs text-faint">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="font-cond text-[24px] font-semibold leading-tight tnum text-paper">{value}</span>
        {unit && <span className="text-2xs text-mute">{unit}</span>}
      </div>
      {hint && <div className="text-2xs text-mute">{hint}</div>}
    </div>
  )
}

type P = { t: number; v: number; zero: boolean }

function EventChart({ e }: { e: HampelEvent }) {
  const [ref, width] = useWidth<HTMLDivElement>(600)
  const iw = Math.max(200, width - M.left - M.right)
  const ih = H - M.top - M.bottom
  const times = e.context.hours.map((s) => Date.parse(s))
  const evT = Date.parse(e.hour_start)

  const raw: P[] = []
  const clean: P[] = []
  e.context.raw.forEach((v, i) => {
    if (v != null) raw.push({ t: times[i], v: Math.max(1, v), zero: v < 1 })
  })
  e.context.clean.forEach((v, i) => {
    if (v != null) clean.push({ t: times[i], v: Math.max(1, v), zero: v < 1 })
  })
  const vmax = Math.max(CAP * 2, ...raw.map((p) => p.v), ...clean.map((p) => p.v))
  const vmin = Math.min(10, ...raw.filter((p) => !p.zero).map((p) => p.v), ...clean.map((p) => p.v))
  const x = scaleLinear().domain([Math.min(...times), Math.max(...times) + 3600e3]).range([0, iw])
  const y = scaleLog().domain([Math.max(1, vmin * 0.7), vmax * 1.6]).range([ih, 0]).clamp(true)
  const xc = (t: number) => x(t + 1800e3)
  const ln = line<P>()
    .x((p) => xc(p.t))
    .y((p) => y(p.v))
  const decades = [1, 10, 100, 1000, 10000, 100000, 1000000].filter((v) => v >= y.domain()[0] && v <= y.domain()[1])

  const midnights = e.context.hours.map((iso, i) => ({ iso, t: times[i] })).filter((m) => m.iso.slice(11, 13) === '00')

  return (
    <div ref={ref}>
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[13px] text-paper">
          {longStamp(e.hour_start)} <span className="text-mute">· raw reading</span>{' '}
          <span className="font-mono tnum">{numFmt.format(e.raw)}</span>
          <span className="text-mute"> replaced by </span>
          <span className="font-mono tnum">{numFmt.format(e.clean)}</span>
        </span>
        <span className="text-2xs text-faint">log scale · µg/m³ · IST</span>
      </div>
      <svg width={width} height={H} role="img" aria-label="Raw versus cleaned readings around the corrected hour">
        <g transform={`translate(${M.left},${M.top})`}>
          {decades.map((v) => (
            <g key={v}>
              <line x1={0} x2={iw} y1={y(v)} y2={y(v)} stroke={C.grid} />
              <text x={-10} y={y(v)} dy="0.32em" textAnchor="end" className="font-mono" fontSize={11} fill={C.faint}>
                {logTick(v)}
              </text>
            </g>
          ))}
          {midnights.map((m) => (
            <g key={m.t}>
              <line x1={x(m.t)} x2={x(m.t)} y1={0} y2={ih} stroke={C.grid} />
              <text x={x(m.t) + 4} y={ih + 18} className="font-mono" fontSize={11} fill={C.mute}>
                {dayLabel(m.iso)}
              </text>
            </g>
          ))}
          <line x1={0} x2={iw} y1={y(CAP)} y2={y(CAP)} stroke="rgba(236,231,220,0.4)" strokeDasharray="5 4" />
          <text x={iw - 4} y={y(CAP) - 6} textAnchor="end" fontSize={11} fill={C.mute}>
            Hard cap · {CAP}
          </text>

          <line x1={xc(evT)} x2={xc(evT)} y1={0} y2={ih} stroke={C.paper} strokeOpacity={0.35} />

          <path d={ln(raw) ?? ''} fill="none" stroke={C.mute} strokeOpacity={0.6} strokeDasharray="3 3" />
          {raw.map((p) => (
            <circle
              key={`r${p.t}`}
              cx={xc(p.t)}
              cy={y(p.v)}
              r={p.v > CAP || p.zero ? 2.8 : 2}
              fill={p.v > CAP || p.zero ? C.ink2 : C.mute}
              stroke={p.v > CAP || p.zero ? C.paper : 'none'}
              strokeWidth={1.1}
            >
              <title>{p.zero ? 'Raw reading 0 (shown at 1 on the log scale)' : `Raw ${numFmt.format(p.v)}`}</title>
            </circle>
          ))}
          <path d={ln(clean) ?? ''} fill="none" stroke={C.paper} strokeWidth={1.75} strokeLinejoin="round" />
          {clean
            .filter((p) => p.t === evT)
            .map((p) => (
              <circle key="ev" cx={xc(p.t)} cy={y(p.v)} r={4} fill={C.paper} stroke={C.ink2} strokeWidth={2} />
            ))}
        </g>
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 pl-[52px] text-2xs text-faint">
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="6" aria-hidden>
            <line x1="0" x2="16" y1="3" y2="3" stroke={C.paper} strokeWidth={1.75} />
          </svg>
          Cleaned (used for training)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="6" aria-hidden>
            <line x1="0" x2="16" y1="3" y2="3" stroke={C.mute} strokeDasharray="3 3" />
          </svg>
          Raw sensor reading
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="8" height="8" aria-hidden>
            <circle cx="4" cy="4" r="2.8" fill={C.ink2} stroke={C.paper} />
          </svg>
          Above the cap or zero
        </span>
      </div>
    </div>
  )
}
