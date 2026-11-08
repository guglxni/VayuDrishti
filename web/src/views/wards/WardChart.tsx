import { scaleLinear } from 'd3-scale'
import { area, line } from 'd3-shape'
import { useId, useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react'
import type { Num, Window } from '../../data/types'
import { grapStage, pmColor } from '../../lib/aqi'
import { fmt } from '../../lib/series'
import { istParts } from '../../lib/stations'
import { PmGradient } from './parts'
import { useSize } from './util'

const BEFORE = 24
const AFTER = 72
const HEIGHT = 228
const M = { top: 22, right: 62, bottom: 28, left: 34 }

const RULES = [
  { v: 60, label: 'Stage I' },
  { v: 90, label: 'Stage II' },
  { v: 120, label: 'Stage III' },
]

export function WardChart({
  series,
  hours,
  hour,
  windows,
  onPick,
}: {
  series: Num[]
  hours: string[]
  hour: number
  windows: Window[]
  onPick: (i: number) => void
}) {
  const [ref, { width }] = useSize<HTMLDivElement>()
  const gid = useId().replace(/:/g, '')
  const [hover, setHover] = useState<number | null>(null)

  const lo = Math.max(0, hour - BEFORE)
  const hi = Math.min(hours.length - 1, hour + AFTER)
  const slice = useMemo(() => series.slice(lo, hi + 1), [series, lo, hi])
  const vals = slice.filter((v): v is number => v != null && Number.isFinite(v))
  const vmax = Math.max(132, (vals.length ? Math.max(...vals) : 0) * 1.1)

  const innerW = Math.max(10, width - M.left - M.right)
  const x = scaleLinear().domain([lo, hi]).range([M.left, M.left + innerW])
  const y = scaleLinear().domain([0, vmax]).range([HEIGHT - M.bottom, M.top])

  const defined = (v: Num) => v != null && Number.isFinite(v)
  const linePath = line<Num>()
    .defined(defined)
    .x((_, k) => x(lo + k))
    .y((v) => y(v as number))(slice)
  const areaPath = area<Num>()
    .defined(defined)
    .x((_, k) => x(lo + k))
    .y0(y(0))
    .y1((v) => y(v as number))(slice)

  const ticks: { i: number; major: boolean }[] = []
  for (let i = lo; i <= hi; i++) {
    const hh = Number(hours[i]?.slice(11, 13))
    if (hh === 0) ticks.push({ i, major: true })
    else if (hh === 12 && innerW > 420) ticks.push({ i, major: false })
  }

  const scored = windows.find((w) => w.id === 'private')
  const scoredSpan = scored && scored.end >= lo && scored.start <= hi ? [Math.max(lo, scored.start), Math.min(hi, scored.end)] : null

  const cur = series[hour]
  const hv = hover != null ? series[hover] : null

  function indexFromEvent(e: MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    return Math.max(lo, Math.min(hi, Math.round(x.invert(px))))
  }

  function onKey(e: KeyboardEvent<SVGSVGElement>) {
    const step = e.shiftKey ? 6 : 1
    if (e.key === 'ArrowRight') onPick(hour + step)
    else if (e.key === 'ArrowLeft') onPick(hour - step)
    else return
    e.preventDefault()
  }

  const tipW = 118
  const tipX = hover != null ? Math.min(x(hover) + 8, M.left + innerW - tipW) : 0
  const tipFlip = hover != null && x(hover) + 8 + tipW > M.left + innerW

  return (
    <div ref={ref} className="relative w-full select-none" style={{ height: HEIGHT }}>
      {width > 0 && (
        <svg
          width={width}
          height={HEIGHT}
          role="slider"
          tabIndex={0}
          aria-label="Ward PM2.5, 24 hours before to 72 hours after the selected hour. Click or use arrow keys to change the hour."
          aria-valuemin={0}
          aria-valuemax={hours.length - 1}
          aria-valuenow={hour}
          aria-valuetext={`${istParts(hours[hour]).short}, ${fmt(cur, 0)} µg/m³`}
          className="block cursor-crosshair focus-visible:outline-none"
          onPointerMove={(e) => setHover(indexFromEvent(e))}
          onPointerLeave={() => setHover(null)}
          onClick={(e) => onPick(indexFromEvent(e))}
          onKeyDown={onKey}
        >
          <defs>
            <PmGradient id={`${gid}-g`} yAtZero={y(0)} yAtMax={y(vmax)} vmax={vmax} />
          </defs>

          {/* Earlier hours */}
          <rect x={x(lo)} y={M.top} width={Math.max(0, x(hour) - x(lo))} height={HEIGHT - M.top - M.bottom} fill="#ECE7DC" fillOpacity={0.025} />

          {scoredSpan && (
            <g>
              <rect
                x={x(scoredSpan[0])}
                y={M.top}
                width={Math.max(1, x(scoredSpan[1]) - x(scoredSpan[0]))}
                height={HEIGHT - M.top - M.bottom}
                fill="#ECE7DC"
                fillOpacity={0.045}
              />
              <line x1={x(scoredSpan[0])} x2={x(scoredSpan[1])} y1={M.top} y2={M.top} stroke="#ECE7DC" strokeOpacity={0.35} />
              <text x={x(scoredSpan[0]) + 4} y={M.top - 6} className="fill-mute font-mono" fontSize={9.5} letterSpacing="0.06em">
                SCORED WINDOW
              </text>
            </g>
          )}

          {/* Y axis */}
          {[0, 30, 60, 90, 120].filter((v) => v <= vmax).map((v) => (
            <text key={v} x={M.left - 8} y={y(v)} dy="0.32em" textAnchor="end" className="fill-faint font-cond tnum" fontSize={10.5}>
              {v}
            </text>
          ))}
          <line x1={M.left} x2={M.left + innerW} y1={y(0)} y2={y(0)} stroke="#ECE7DC" strokeOpacity={0.14} />

          {/* GRAP thresholds */}
          {RULES.map((r) => {
            const c = grapStage(r.v).color
            return (
              <g key={r.v}>
                <line x1={M.left} x2={M.left + innerW} y1={y(r.v)} y2={y(r.v)} stroke={c} strokeOpacity={0.32} strokeDasharray="3 4" />
                <text x={M.left + innerW + 6} y={y(r.v)} dy="0.32em" fill={c} fillOpacity={0.85} fontSize={10.5}>
                  {r.label}
                </text>
              </g>
            )
          })}

          {/* Day ticks */}
          {ticks.map(({ i, major }) => {
            const p = istParts(hours[i])
            return (
              <g key={i}>
                <line
                  x1={x(i)}
                  x2={x(i)}
                  y1={M.top}
                  y2={HEIGHT - M.bottom}
                  stroke="#ECE7DC"
                  strokeOpacity={major ? 0.08 : 0.035}
                />
                <text
                  x={x(i)}
                  y={HEIGHT - M.bottom + 16}
                  textAnchor={major ? 'start' : 'middle'}
                  dx={major ? 3 : 0}
                  className={major ? 'fill-mute' : 'fill-faint font-mono'}
                  fontSize={major ? 11 : 9.5}
                >
                  {major ? `${p.day} ${p.date}` : p.time}
                </text>
              </g>
            )
          })}

          {areaPath && <path d={areaPath} fill={`url(#${gid}-g)`} fillOpacity={0.1} />}
          {linePath && <path d={linePath} fill="none" stroke={`url(#${gid}-g)`} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />}
          {vals.length === 0 && (
            <text x={M.left + innerW / 2} y={(M.top + HEIGHT - M.bottom) / 2} textAnchor="middle" className="fill-mute" fontSize={12}>
              No ward estimate in this period
            </text>
          )}

          {/* Selected hour */}
          <line x1={x(hour)} x2={x(hour)} y1={M.top - 2} y2={HEIGHT - M.bottom} stroke="#ECE7DC" strokeOpacity={0.8} />
          <text
            x={x(hour)}
            y={M.top - 6}
            textAnchor={x(hour) > M.left + innerW - 40 ? 'end' : 'middle'}
            className="fill-paper font-mono"
            fontSize={10}
          >
            {istParts(hours[hour]).time}
          </text>
          {cur != null && <circle cx={x(hour)} cy={y(cur)} r={4.5} fill={pmColor(cur)} stroke="#0A1119" strokeWidth={2} />}

          {/* Hover */}
          {hover != null && hover !== hour && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={HEIGHT - M.bottom} stroke="#ECE7DC" strokeOpacity={0.3} strokeDasharray="2 2" />
              {hv != null && <circle cx={x(hover)} cy={y(hv)} r={3} fill={pmColor(hv)} />}
            </g>
          )}
          {hover != null && (
            <g pointerEvents="none" transform={`translate(${tipFlip ? x(hover) - tipW - 8 : tipX}, ${M.top + 6})`}>
              <rect width={tipW} height={38} rx={7} fill="#16212C" stroke="#ECE7DC" strokeOpacity={0.14} />
              <text x={9} y={15} className="fill-mute font-mono" fontSize={10}>
                {istParts(hours[hover]).day} {istParts(hours[hover]).short}
              </text>
              <text x={9} y={31} fontSize={13} className="font-cond tnum" fontWeight={600} fill={pmColor(hv)}>
                {fmt(hv, 1)}
                <tspan className="fill-mute" fontSize={10} fontWeight={400} dx={4}>
                  µg/m³ · {grapStage(hv).level < 0 ? 'no data' : grapStage(hv).stage}
                </tspan>
              </text>
            </g>
          )}
        </svg>
      )}
    </div>
  )
}
