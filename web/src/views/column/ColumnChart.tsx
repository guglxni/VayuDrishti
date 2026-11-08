import { useMemo, useRef } from 'react'
import { scaleLinear } from 'd3-scale'
import { area, curveMonotoneX, line } from 'd3-shape'
import type { TwinData } from '../../data/types'
import { pmColor } from '../../lib/aqi'
import { solarElevation } from '../../lib/physics'
import { stationPM } from '../../lib/series'
import { istParts } from '../../lib/stations'
import { useTwin } from '../../state/store'
import { fmtInt, stationMet } from './derive'

const W = 308
const H = 128
const M = { top: 8, right: 30, bottom: 18, left: 34 }
const BEFORE = 24
const AFTER = 48

interface Pt {
  i: number
  pblh: number | null
  pm: number | null
  night: boolean
}

/** Mixing height (area) and PM2.5 (line) from 24 h before to 48 h after the current hour; click or drag to scrub. */
export function ColumnChart({ data, stationId, hour }: { data: TwinData; stationId: number; hour: number }) {
  const setHour = useTwin((s) => s.setHour)
  const setPlaying = useTwin((s) => s.setPlaying)
  const svg = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)
  const n = data.forecasts.hours.length
  const lo = hour - BEFORE
  const hi = hour + AFTER

  const pts = useMemo(() => {
    const out: Pt[] = []
    for (let i = Math.max(0, lo); i <= Math.min(n - 1, hi); i++) {
      out.push({
        i,
        pblh: stationMet(data, stationId, i).pblh,
        pm: stationPM(data, stationId, i),
        night: solarElevation(data.forecasts.hours[i]) < 0,
      })
    }
    return out
  }, [data, stationId, lo, hi, n])

  const x = scaleLinear().domain([lo, hi]).range([M.left, W - M.right])
  const maxH = Math.max(1000, ...pts.map((p) => p.pblh ?? 0))
  const yH = scaleLinear()
    .domain([0, maxH * 1.08])
    .range([H - M.bottom, M.top])
    .nice(4)
  const maxPm = Math.max(80, ...pts.map((p) => p.pm ?? 0))
  const yP = scaleLinear()
    .domain([0, maxPm * 1.1])
    .range([H - M.bottom, M.top])
    .nice(4)

  const areaPath = area<Pt>()
    .defined((p) => p.pblh != null)
    .x((p) => x(p.i))
    .y0(H - M.bottom)
    .y1((p) => yH(p.pblh!))
    .curve(curveMonotoneX)(pts)
  const topPath = line<Pt>()
    .defined((p) => p.pblh != null)
    .x((p) => x(p.i))
    .y((p) => yH(p.pblh!))
    .curve(curveMonotoneX)(pts)

  const nightBands: [number, number][] = []
  for (const p of pts) {
    if (!p.night) continue
    const last = nightBands[nightBands.length - 1]
    if (last && last[1] === p.i - 1) last[1] = p.i
    else nightBands.push([p.i, p.i])
  }
  const midnights = pts.filter((p) => data.forecasts.hours[p.i].slice(11, 13) === '00')
  const cur = pts.find((p) => p.i === hour)

  const scrub = (clientX: number) => {
    const el = svg.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const px = ((clientX - r.left) / r.width) * W
    setHour(Math.round(x.invert(Math.max(M.left, Math.min(W - M.right, px)))))
  }

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-2xs">
        <div className="flex items-center gap-3 text-mute">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-3 rounded-[2px] border border-wind/60 bg-wind/20" />
            Mixing height
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded-full" style={{ background: 'linear-gradient(90deg,#3DAE6B,#EFD143,#E2483F)' }} />
            PM2.5
          </span>
        </div>
        <span className="font-mono text-[10px] text-faint">−24 h … +48 h</span>
      </div>
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full cursor-ew-resize touch-none select-none"
        role="slider"
        aria-label="Scrub time"
        aria-valuemin={0}
        aria-valuemax={n - 1}
        aria-valuenow={hour}
        onPointerDown={(e) => {
          dragging.current = true
          e.currentTarget.setPointerCapture(e.pointerId)
          setPlaying(false)
          scrub(e.clientX)
        }}
        onPointerMove={(e) => dragging.current && scrub(e.clientX)}
        onPointerUp={(e) => {
          dragging.current = false
          e.currentTarget.releasePointerCapture(e.pointerId)
        }}
        onPointerCancel={() => (dragging.current = false)}
      >
        {nightBands.map(([a, b]) => (
          <rect
            key={a}
            x={x(a - 0.5)}
            y={M.top}
            width={Math.max(0, x(b + 0.5) - x(a - 0.5))}
            height={H - M.top - M.bottom}
            fill="rgb(0 0 0 / 0.22)"
          />
        ))}
        {yH.ticks(3).map((t) => (
          <g key={`h${t}`}>
            <line x1={M.left} x2={W - M.right} y1={yH(t)} y2={yH(t)} stroke="rgb(236 231 220 / 0.06)" />
            <text x={M.left - 4} y={yH(t)} dy="0.32em" textAnchor="end" className="fill-wind/60 font-mono text-[8.5px]">
              {t >= 1000 ? `${t / 1000}k` : t}
            </text>
          </g>
        ))}
        {yP.ticks(3).map((t) => (
          <text key={`p${t}`} x={W - M.right + 4} y={yP(t)} dy="0.32em" className="fill-paper/45 font-mono text-[8.5px]">
            {t}
          </text>
        ))}
        <text x={M.left - 4} y={H - 4} textAnchor="end" className="fill-wind/50 font-mono text-[8px]">
          m
        </text>
        <text x={W - M.right + 4} y={H - 4} className="fill-paper/40 font-mono text-[8px]">
          µg
        </text>
        {areaPath && <path d={areaPath} fill="rgb(207 227 242 / 0.13)" />}
        {topPath && <path d={topPath} fill="none" stroke="rgb(207 227 242 / 0.6)" strokeWidth={1} />}
        {pts.slice(1).map((p, k) => {
          const a = pts[k]
          if (a.pm == null || p.pm == null) return null
          return (
            <line
              key={p.i}
              x1={x(a.i)}
              y1={yP(a.pm)}
              x2={x(p.i)}
              y2={yP(p.pm)}
              stroke={pmColor((a.pm + p.pm) / 2)}
              strokeWidth={1.75}
              strokeLinecap="round"
            />
          )
        })}
        {midnights.map((p) => (
          <g key={`m${p.i}`}>
            <line x1={x(p.i)} x2={x(p.i)} y1={H - M.bottom} y2={H - M.bottom + 3} stroke="rgb(236 231 220 / 0.3)" />
            <text x={x(p.i)} y={H - 4} textAnchor="middle" className="fill-faint font-mono text-[8.5px]">
              {istParts(data.forecasts.hours[p.i]).date}
            </text>
          </g>
        ))}
        <line x1={x(hour)} x2={x(hour)} y1={M.top - 2} y2={H - M.bottom} stroke="#ECE7DC" strokeWidth={1} />
        {cur?.pblh != null && <circle cx={x(hour)} cy={yH(cur.pblh)} r={2.75} fill="#0A1119" stroke="#CFE3F2" strokeWidth={1.25} />}
        {cur?.pm != null && <circle cx={x(hour)} cy={yP(cur.pm)} r={3} fill={pmColor(cur.pm)} stroke="#0A1119" strokeWidth={1} />}
        <rect x={0} y={0} width={W} height={H} fill="transparent" />
      </svg>
      {cur && (
        <div className="mt-1 flex justify-between font-mono text-[10px] text-faint tnum">
          <span>Shaded: night</span>
          <span>
            {fmtInt(cur.pblh)} m · {cur.pm != null ? Math.round(cur.pm) : '—'} µg/m³
          </span>
        </div>
      )}
    </div>
  )
}
