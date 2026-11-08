import { memo, useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react'
import { area, curveMonotoneX, line } from 'd3-shape'
import { scaleLinear } from 'd3-scale'
import { useData, useTwin } from '../state/store'
import { NAAQS_24H, pmColor } from '../lib/aqi'
import { fmt, stationPM } from '../lib/series'
import { istParts, stationName, TARGET_IDS } from '../lib/stations'
import type { TwinData } from '../data/types'

const BEFORE = 24
const AFTER = 48
const DETAIL_H = 76
const LANE_H = 4
const LANE_GAP = 1
const RIBBON_TOP = DETAIL_H + 34
const RIBBON_H = TARGET_IDS.length * (LANE_H + LANE_GAP) - LANE_GAP

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(800)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Four stations × 960 hours, each hour a sliver coloured by forecast PM2.5. */
const Ribbon = memo(function Ribbon({ data, width, selected }: { data: TwinData; width: number; selected: number }) {
  const n = data.forecasts.hours.length
  const cw = width / n
  return (
    <g>
      {TARGET_IDS.map((id, lane) => {
        const y = RIBBON_TOP + lane * (LANE_H + LANE_GAP)
        const rects = []
        for (let i = 0; i < n; i++) {
          const v = stationPM(data, id, i)
          rects.push(<rect key={i} x={i * cw} y={y} width={cw + 0.6} height={LANE_H} fill={pmColor(v)} />)
        }
        return (
          <g key={id} opacity={id === selected ? 1 : 0.62}>
            {rects}
          </g>
        )
      })}
    </g>
  )
})

export function Timeline() {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const setHour = useTwin((s) => s.setHour)
  const playing = useTwin((s) => s.playing)
  const setPlaying = useTwin((s) => s.setPlaying)
  const stationId = useTwin((s) => s.stationId)
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hoverI, setHoverI] = useState<number | null>(null)

  const hours = data.forecasts.hours
  const n = hours.length
  const fc = data.forecasts.stations[String(stationId)]
  const st = data.stations.stations.find((s) => s.id === stationId)

  const x0 = Math.max(0, Math.min(n - 1 - BEFORE - AFTER, hour - BEFORE))
  const x1 = x0 + BEFORE + AFTER
  const xd = useMemo(() => scaleLinear().domain([x0, x1]).range([0, width]), [x0, x1, width])
  const xr = useMemo(() => scaleLinear().domain([0, n - 1]).range([0, width]), [n, width])

  const detail = useMemo(() => {
    const idx: number[] = []
    for (let i = x0; i <= x1; i++) idx.push(i)
    const ens = idx.map((i) => fc.models.ensemble[i])
    const lo = idx.map((i) => fc.deciles[i]?.[0] ?? null)
    const hi = idx.map((i) => fc.deciles[i]?.[8] ?? null)
    const vals = [...ens, ...hi, NAAQS_24H + 10].filter((v): v is number => v != null)
    const ymax = Math.max(...vals) * 1.08
    const y = scaleLinear().domain([0, ymax]).range([DETAIL_H, 6])
    const band = area<number>()
      .defined((i) => fc.deciles[i]?.[0] != null)
      .x((i) => xd(i))
      .y0((i) => y(fc.deciles[i]![0]!))
      .y1((i) => y(fc.deciles[i]![8]!))
      .curve(curveMonotoneX)(idx)
    const ln = line<number>()
      .defined((i) => fc.models.ensemble[i] != null)
      .x((i) => xd(i))
      .y((i) => y(fc.models.ensemble[i]!))
      .curve(curveMonotoneX)(idx)
    const nights: [number, number][] = []
    let start: number | null = null
    for (const i of idx) {
      const sw = data.wind.city.sw[i] ?? 0
      if (sw <= 1 && start == null) start = i
      if (sw > 1 && start != null) {
        nights.push([start, i])
        start = null
      }
    }
    if (start != null) nights.push([start, x1])
    const obs = idx.filter((i) => fc.obs[i] != null).map((i) => ({ i, v: fc.obs[i]! }))
    return { idx, y, band, ln, nights, obs, lo, hi }
  }, [fc, x0, x1, xd, data])

  const days = useMemo(() => {
    const out: number[] = []
    for (let i = x0; i <= x1; i++) if (hours[i]?.slice(11, 13) === '00') out.push(i)
    return out
  }, [hours, x0, x1])

  const weeks = useMemo(() => {
    const out: number[] = []
    for (let i = 0; i < n; i++) if (hours[i].slice(11, 13) === '00' && new Date(hours[i]).getUTCDay() === 1) out.push(i)
    return out
  }, [hours, n])

  const win = (id: string) => data.forecasts.windows.find((w) => w.id === id)
  const validation = win('validation')
  const priv = win('private')

  const dragging = useRef<'detail' | 'ribbon' | null>(null)
  const toIndex = (e: RPointerEvent<SVGSVGElement>, zone: 'detail' | 'ribbon') => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    return Math.round(zone === 'detail' ? xd.invert(px) : xr.invert(px))
  }
  const onDown = (e: RPointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const zone = e.clientY - r.top > DETAIL_H + 12 ? 'ribbon' : 'detail'
    dragging.current = zone
    e.currentTarget.setPointerCapture(e.pointerId)
    setPlaying(false)
    setHour(toIndex(e, zone))
  }
  const onMove = (e: RPointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    if (dragging.current) {
      setHour(toIndex(e, dragging.current))
      return
    }
    const zone = e.clientY - r.top > DETAIL_H + 12 ? 'ribbon' : 'detail'
    setHoverI(Math.max(0, Math.min(n - 1, toIndex(e, zone))))
  }
  const onUp = () => {
    dragging.current = null
  }

  const when = istParts(hours[hour])
  const block = data.forecasts.blocks[hour]
  const lead = data.forecasts.horizon[hour]
  const status =
    block === 'validation'
      ? 'Hindcast · checked against observations'
      : block === 'private'
        ? `Scored window · ${Math.round(lead / 24)} days after the last observation`
        : `Forecast · ${lead < 48 ? `${lead} h` : `${Math.round(lead / 24)} days`} after the last observation`

  const pm = stationPM(data, stationId, hour)
  const hv = hoverI != null ? hoverI : null
  const H = RIBBON_TOP + RIBBON_H + 18

  return (
    <section
      className="panel absolute bottom-4 left-5 right-5 z-20 flex items-stretch gap-5 px-4 py-3"
      aria-label="Timeline"
    >
      <div className="flex w-[218px] shrink-0 flex-col justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? 'Pause' : 'Play'}
            title="Play / pause (space)"
            className="flex h-10 w-10 items-center justify-center rounded-full bg-paper text-ink transition-transform hover:scale-105"
          >
            {playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" className="ml-0.5" />}
          </button>
          <button
            onClick={() => setHour(hour - 1)}
            aria-label="Previous hour"
            title="Previous hour (←, shift for a day)"
            className="flex h-8 w-8 items-center justify-center rounded-full text-mute hover:bg-paper/8 hover:text-paper"
          >
            <ChevronLeft size={17} />
          </button>
          <button
            onClick={() => setHour(hour + 1)}
            aria-label="Next hour"
            title="Next hour (→, shift for a day)"
            className="flex h-8 w-8 items-center justify-center rounded-full text-mute hover:bg-paper/8 hover:text-paper"
          >
            <ChevronRight size={17} />
          </button>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-[15px] font-medium">
              {when.day} {when.date}
            </span>
            <span className="font-cond text-[26px] font-semibold leading-none tnum">{when.time}</span>
            <span className="text-2xs text-faint">IST</span>
          </div>
          <div className="mt-0.5 text-2xs leading-snug text-mute">{status}</div>
        </div>
      </div>

      <div ref={ref} className="relative min-w-0 flex-1">
        <div className="pointer-events-none absolute left-0 top-0 z-10 flex items-baseline gap-2 text-2xs">
          <span className="font-medium text-paper">{st ? stationName(st) : ''}</span>
          <span className="text-faint">PM2.5 forecast with 80% range · 1 day back, 2 days ahead</span>
        </div>
        <div className="pointer-events-none absolute right-0 top-0 z-10 text-right text-2xs text-mute tnum">
          {hv != null ? (
            <>
              {istParts(hours[hv]).day} {istParts(hours[hv]).short}
              {' · '}
              <span style={{ color: pmColor(stationPM(data, stationId, hv)) }}>{fmt(stationPM(data, stationId, hv))}</span> µg/m³
            </>
          ) : (
            <>
              Now <span style={{ color: pmColor(pm) }}>{fmt(pm)}</span> µg/m³
            </>
          )}
        </div>
        <svg
          width={width}
          height={H}
          className="block touch-none select-none"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={() => setHoverI(null)}
          role="slider"
          aria-label="Forecast hour"
          aria-valuemin={0}
          aria-valuemax={n - 1}
          aria-valuenow={hour}
          aria-valuetext={`${when.day} ${when.short} IST`}
        >
          <defs>
            <linearGradient id="tl-line" gradientUnits="userSpaceOnUse" x1={0} x2={width} y1={0} y2={0}>
              {detail.idx.map((i) => (
                <stop key={i} offset={`${((i - x0) / (x1 - x0)) * 100}%`} stopColor={pmColor(fc.models.ensemble[i])} />
              ))}
            </linearGradient>
          </defs>
          {detail.nights.map(([a, b]) => (
            <rect key={a} x={xd(a)} y={14} width={Math.max(0, xd(b) - xd(a))} height={DETAIL_H - 14} fill="#ece7dc" opacity={0.035} />
          ))}
          <line x1={0} x2={width} y1={detail.y(NAAQS_24H)} y2={detail.y(NAAQS_24H)} stroke="#ece7dc" strokeOpacity={0.25} strokeDasharray="3 4" />
          <text x={width - 2} y={detail.y(NAAQS_24H) - 4} textAnchor="end" className="fill-faint font-mono text-[10px]">
            India standard 60
          </text>
          {detail.band && <path d={detail.band} fill="#ece7dc" opacity={0.09} />}
          {detail.ln && <path d={detail.ln} fill="none" stroke="url(#tl-line)" strokeWidth={2.2} strokeLinecap="round" />}
          {detail.obs.map((o) => (
            <circle key={o.i} cx={xd(o.i)} cy={detail.y(o.v)} r={1.8} fill="#ece7dc" opacity={0.75} />
          ))}
          <line x1={0} x2={width} y1={DETAIL_H} y2={DETAIL_H} stroke="#ece7dc" strokeOpacity={0.1} />
          {days.map((i) => (
            <g key={i}>
              <line x1={xd(i)} x2={xd(i)} y1={14} y2={DETAIL_H + 4} stroke="#ece7dc" strokeOpacity={0.12} />
              <text x={xd(i) + 4} y={DETAIL_H + 13} className="fill-faint font-mono text-[10px]">
                {istParts(hours[i]).day} {istParts(hours[i]).date}
              </text>
            </g>
          ))}
          {/* cursor on detail */}
          <line x1={xd(hour)} x2={xd(hour)} y1={10} y2={DETAIL_H} stroke="#ece7dc" strokeWidth={1.2} />
          {fc.models.ensemble[hour] != null && (
            <circle
              cx={xd(hour)}
              cy={detail.y(fc.models.ensemble[hour]!)}
              r={4.5}
              fill={pmColor(fc.models.ensemble[hour])}
              stroke="#0a1119"
              strokeWidth={2}
            />
          )}
          {hv != null && hv >= x0 && hv <= x1 && !dragging.current && (
            <line x1={xd(hv)} x2={xd(hv)} y1={14} y2={DETAIL_H} stroke="#ece7dc" strokeOpacity={0.3} />
          )}

          {/* ribbon: whole period */}
          {validation && (
            <g>
              <line x1={xr(validation.start)} x2={xr(validation.end + 1)} y1={RIBBON_TOP - 5} y2={RIBBON_TOP - 5} stroke="#ece7dc" strokeOpacity={0.3} />
              <text x={xr(validation.start)} y={RIBBON_TOP - 9} className="fill-faint text-[10px]">
                Checked against observations
              </text>
            </g>
          )}
          <g>
            <line x1={xr(240)} x2={xr(240)} y1={RIBBON_TOP - 12} y2={RIBBON_TOP + RIBBON_H} stroke="#ece7dc" strokeOpacity={0.5} />
            <text x={xr(240) + 4} y={RIBBON_TOP - 9} className="fill-mute text-[10px]">
              Forecast from 22 Oct
            </text>
          </g>
          <Ribbon data={data} width={width} selected={stationId} />
          {priv && (
            <g>
              <rect
                x={xr(priv.start)}
                y={RIBBON_TOP - 2}
                width={xr(priv.end + 1) - xr(priv.start)}
                height={RIBBON_H + 4}
                fill="none"
                stroke="#ece7dc"
                strokeWidth={1.2}
                rx={2}
              />
              <text x={xr(priv.start) + (xr(priv.end + 1) - xr(priv.start)) / 2} y={RIBBON_TOP - 6} textAnchor="middle" className="fill-paper text-[10px] font-medium">
                Scored 11–12 Nov
              </text>
            </g>
          )}
          {/* lens showing the detail window */}
          <rect
            x={xr(x0)}
            y={RIBBON_TOP - 3}
            width={xr(x1) - xr(x0)}
            height={RIBBON_H + 6}
            fill="#ece7dc"
            fillOpacity={0.08}
            stroke="#ece7dc"
            strokeOpacity={0.35}
            rx={2}
          />
          <line x1={xr(hour)} x2={xr(hour)} y1={RIBBON_TOP - 5} y2={RIBBON_TOP + RIBBON_H + 5} stroke="#ece7dc" strokeWidth={2} />
          {weeks.map((i) => (
            <text key={i} x={xr(i)} y={RIBBON_TOP + RIBBON_H + 14} className="fill-faint font-mono text-[10px]">
              {istParts(hours[i]).date}
            </text>
          ))}
        </svg>
      </div>
    </section>
  )
}
