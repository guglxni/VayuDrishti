import { line } from 'd3-shape'
import { scaleLinear } from 'd3-scale'
import { useId } from 'react'
import type { Num } from '../../data/types'
import { grapStage, NAAQS_24H, pmColor } from '../../lib/aqi'
import { cx } from '../../ui/primitives'

/** GRAP stage pill coloured by the stage's own threshold colour. */
export function GrapChip({ value, className }: { value: number | null; className?: string }) {
  const g = grapStage(value)
  const label = g.level < 0 ? 'No data' : g.stage
  return (
    <span
      className={cx('inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-2xs font-medium whitespace-nowrap', className)}
      style={{ background: `${g.color}1a`, color: g.color, boxShadow: `inset 0 0 0 1px ${g.color}33` }}
      title={g.level < 0 ? 'No ward estimate this hour' : `GRAP ${g.stage}: Watch < 60, Stage I 60–90, Stage II 90–120, Stage III ≥ 120 µg/m³`}
    >
      {label}
    </span>
  )
}

/**
 * Vertical gradient that maps y-pixels to the PM2.5 colour ramp, so a stroke drawn with it
 * is coloured by its own value at every point.
 */
export function PmGradient({ id, yAtZero, yAtMax, vmax }: { id: string; yAtZero: number; yAtMax: number; vmax: number }) {
  const stops: number[] = []
  for (let v = 0; v <= vmax; v += Math.max(2, vmax / 40)) stops.push(v)
  stops.push(vmax)
  return (
    <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="0" x2="0" y1={yAtZero} y2={yAtMax}>
      {stops.map((v, k) => (
        <stop key={k} offset={v / vmax} stopColor={pmColor(v)} />
      ))}
    </linearGradient>
  )
}

/** Next-72 h sparkline with a faint NAAQS reference. */
export function Sparkline({ series, from, span = 72, width = 72, height = 22 }: { series: Num[]; from: number; span?: number; width?: number; height?: number }) {
  const gid = useId().replace(/:/g, '')
  const slice = series.slice(from, from + span)
  const vals = slice.filter((v): v is number => v != null && Number.isFinite(v))
  if (vals.length < 2) {
    return (
      <svg width={width} height={height} aria-hidden className="shrink-0">
        <line x1={0} x2={width} y1={height / 2} y2={height / 2} stroke="currentColor" strokeOpacity={0.12} strokeDasharray="2 3" />
      </svg>
    )
  }
  const vmax = Math.max(90, ...vals) * 1.05
  const x = scaleLinear().domain([0, span - 1]).range([1, width - 1])
  const y = scaleLinear().domain([0, vmax]).range([height - 1.5, 1.5])
  const path = line<Num>()
    .defined((v) => v != null && Number.isFinite(v))
    .x((_, i) => x(i))
    .y((v) => y(v as number))(slice)
  return (
    <svg width={width} height={height} aria-hidden className="shrink-0 overflow-visible">
      <defs>
        <PmGradient id={gid} yAtZero={y(0)} yAtMax={y(vmax)} vmax={vmax} />
      </defs>
      <line x1={0} x2={width} y1={y(NAAQS_24H)} y2={y(NAAQS_24H)} stroke="#ECE7DC" strokeOpacity={0.18} strokeDasharray="2 2" />
      <path d={path ?? ''} fill="none" stroke={`url(#${gid})`} strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}
