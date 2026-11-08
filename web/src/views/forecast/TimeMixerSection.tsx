import { useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react'
import { scaleLinear } from 'd3-scale'
import { area, line } from 'd3-shape'
import type { Forecasts, Num } from '../../data/types'
import { fmt } from '../../lib/series'
import { cx } from '../../ui/primitives'
import { C, dayLabel, SectionHead, useWidth } from './shared'

const LEFT = 50
const RIGHT = 18
const TOP_Y = 30
const TOP_H = 146
const BOTTOM_Y = 238
const BOTTOM_H = 88
const HEIGHT = 354
const WIND = '#CFE3F2'
const TRAFFIC = '#F0973A'

type Pt = { i: number; v: number }

function points(series: Num[] | undefined, a: number, b: number): (Pt | null)[] {
  const out: (Pt | null)[] = []
  for (let i = a; i <= b; i++) {
    const value = series?.[i]
    out.push(value == null || !Number.isFinite(value) ? null : { i, v: value })
  }
  return out
}

const TOP_LEGEND: { label: string; color: string; width: number; opacity?: number; dash?: string }[] = [
  { label: 'Ensemble', color: C.paper, width: 2.2 },
  { label: 'L_s(t) · 24h synoptic', color: C.paper, width: 1.4, opacity: 0.58 },
  { label: 'M1 · tanh-MOS CAMS', color: WIND, width: 1.4 },
  { label: 'Raw CAMS', color: C.faint, width: 1.4, dash: '4 4' },
] as const

export function TimeMixerSection({
  fc,
  sid,
  range,
  hour,
  setHour,
}: {
  fc: Forecasts
  sid: number
  range: [number, number]
  hour: number
  setHour: (hour: number) => void
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>(1000)
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const st = fc.stations[String(sid)]
  const [a, b] = range
  const iw = Math.max(80, width - LEFT - RIGHT)
  const span = b - a + 1

  const x = useMemo(() => scaleLinear().domain([a, b + 1]).range([0, iw]), [a, b, iw])
  const cx = (i: number) => x(i + 0.5)

  const yTop = useMemo(() => {
    let max = 10
    for (const series of [st.level, st.cams, st.models.m1, st.models.ensemble]) {
      for (let i = a; i <= b; i++) {
        const value = series?.[i]
        if (value != null && Number.isFinite(value)) max = Math.max(max, value)
      }
    }
    return scaleLinear().domain([0, max * 1.06]).range([TOP_H, 0]).nice()
  }, [a, b, st])

  const yDiurnal = useMemo(() => {
    let max = 1
    for (let i = a; i <= b; i++) {
      const value = st.diurnal[i]
      if (value != null && Number.isFinite(value)) max = Math.max(max, Math.abs(value))
    }
    return scaleLinear().domain([-max * 1.12, max * 1.12]).range([BOTTOM_H, 0]).nice()
  }, [a, b, st.diurnal])

  const paths = useMemo(() => {
    const toLine = line<Pt | null>()
      .defined((point) => point != null)
      .x((point) => cx(point!.i))
      .y((point) => yTop(point!.v))
    const toDiurnalLine = line<Pt | null>()
      .defined((point) => point != null)
      .x((point) => cx(point!.i))
      .y((point) => yDiurnal(point!.v))
    const toArea = area<Pt | null>()
      .defined((point) => point != null)
      .x((point) => cx(point!.i))
      .y0(yDiurnal(0))
      .y1((point) => yDiurnal(point!.v))
    return {
      level: toLine(points(st.level, a, b)) ?? '',
      cams: toLine(points(st.cams, a, b)) ?? '',
      m1: toLine(points(st.models.m1, a, b)) ?? '',
      ensemble: toLine(points(st.models.ensemble, a, b)) ?? '',
      diurnal: toDiurnalLine(points(st.diurnal, a, b)) ?? '',
      diurnalArea: toArea(points(st.diurnal, a, b)) ?? '',
    }
  }, [a, b, st, x, yTop, yDiurnal])

  const dayTicks = useMemo(() => {
    const days: { i: number; label: string }[] = []
    for (let i = a; i <= b; i++) {
      if (fc.hours[i].slice(11, 13) === '00') days.push({ i, label: dayLabel(fc.hours[i]) })
    }
    const pxPerDay = (iw / span) * 24
    const step = Math.max(1, Math.ceil(64 / Math.max(1, pxPerDay)))
    return days.filter((_, i) => i % step === 0)
  }, [a, b, fc.hours, iw, span])

  const focusIdx = hoverIdx ?? (hour >= a && hour <= b ? hour : null)
  const readIdx = focusIdx ?? a
  const topTicks = yTop.ticks(4)
  const diurnalTicks = yDiurnal.ticks(4)
  const zeroY = yDiurnal(0)

  function idxFromEvent(event: MouseEvent<SVGRectElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return Math.max(a, Math.min(b, Math.floor(x.invert(event.clientX - rect.left))))
  }

  function onKey(event: KeyboardEvent<SVGSVGElement>) {
    const step = event.shiftKey ? 24 : 1
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') setHoverIdx(null)
    if (event.key === 'ArrowRight') setHour(Math.min(fc.hours.length - 1, hour + step))
    else if (event.key === 'ArrowLeft') setHour(Math.max(0, hour - step))
    else return
    event.preventDefault()
    event.stopPropagation()
  }

  const interactionProps = {
    onPointerMove: (event: MouseEvent<SVGRectElement>) => setHoverIdx(idxFromEvent(event)),
    onPointerLeave: () => setHoverIdx(null),
    onClick: (event: MouseEvent<SVGRectElement>) => setHour(idxFromEvent(event)),
  }

  return (
    <section>
      <SectionHead
        eyebrow="TimeMixer · multi-scale decomposition"
        title="Separate the synoptic baseline from Mumbai’s daily breathing"
        subtitle="The upper chart compares L_s(t) with raw CAMS, its tanh-MOS calibrated M1 prior, and the final ensemble. The lower chart isolates the zero-centred D_s(t) sea-breeze and traffic wave. Both panels share the active forecast range and time cursor."
      />

      <div className="panel-solid relative p-4">
        <div ref={wrapRef} className="min-w-0">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
          <div className="flex flex-wrap gap-x-4 gap-y-1.5">
            {TOP_LEGEND.map((item) => (
              <span key={item.label} className="flex items-center gap-1.5 text-[10px] text-mute">
                <svg width="20" height="7" aria-hidden>
                  <line
                    x1="0"
                    x2="20"
                    y1="3.5"
                    y2="3.5"
                    stroke={item.color}
                    strokeOpacity={item.opacity}
                    strokeWidth={item.width}
                    strokeDasharray={item.dash}
                  />
                </svg>
                {item.label}
              </span>
            ))}
          </div>
          <span className="font-mono text-[10px] text-faint">PM2.5 · µg/m³ · IST hour-start</span>
        </div>

        <svg
          width={width}
          height={HEIGHT}
          role="img"
          tabIndex={0}
          aria-label={`TimeMixer decomposition for station ${sid}. Upper panel: synoptic level, raw and calibrated CAMS, and ensemble. Lower panel: diurnal residual. Click either panel to move the time cursor; use arrow keys for hourly movement.`}
          onKeyDown={onKey}
          className="mt-2 block max-w-full select-none outline-none focus-visible:outline-2 focus-visible:outline-paper/40"
        >
          <defs>
            <clipPath id="tm-top-clip">
              <rect x="0" y="0" width={iw} height={TOP_H} />
            </clipPath>
            <clipPath id="tm-diurnal-clip">
              <rect x="0" y="0" width={iw} height={BOTTOM_H} />
            </clipPath>
            <clipPath id="tm-positive-clip">
              <rect x="0" y="0" width={iw} height={zeroY} />
            </clipPath>
            <clipPath id="tm-negative-clip">
              <rect x="0" y={zeroY} width={iw} height={BOTTOM_H - zeroY} />
            </clipPath>
          </defs>

          <text x={LEFT} y={13} fontSize={11} fontWeight={500} fill={C.paper}>
            Synoptic baseline L_s(t), CAMS calibration & final forecast
          </text>
          <g transform={`translate(${LEFT},${TOP_Y})`}>
            {topTicks.map((tick) => (
              <g key={tick}>
                <line x1="0" x2={iw} y1={yTop(tick)} y2={yTop(tick)} stroke={tick === 0 ? C.gridStrong : C.grid} />
                <text x="-10" y={yTop(tick)} dy="0.32em" textAnchor="end" className="font-mono" fontSize={10} fill={C.faint}>
                  {tick}
                </text>
              </g>
            ))}
            {dayTicks.map((tick) => (
              <line key={tick.i} x1={x(tick.i)} x2={x(tick.i)} y1="0" y2={TOP_H} stroke={C.grid} />
            ))}
            <g clipPath="url(#tm-top-clip)">
              <path d={paths.cams} fill="none" stroke={C.faint} strokeWidth="1.35" strokeDasharray="4 4" />
              <path d={paths.level} fill="none" stroke={C.paper} strokeOpacity="0.58" strokeWidth="1.4" />
              <path d={paths.m1} fill="none" stroke={WIND} strokeWidth="1.4" />
              <path d={paths.ensemble} fill="none" stroke={C.ink} strokeOpacity="0.75" strokeWidth="4" />
              <path d={paths.ensemble} fill="none" stroke={C.paper} strokeWidth="2.2" strokeLinejoin="round" />
            </g>
            {focusIdx != null && (
              <g pointerEvents="none">
                <line x1={cx(focusIdx)} x2={cx(focusIdx)} y1="0" y2={TOP_H} stroke={C.paper} strokeOpacity="0.3" />
                {[
                  { key: 'level', value: st.level[focusIdx], color: C.paper },
                  { key: 'cams', value: st.cams[focusIdx], color: C.faint },
                  { key: 'm1', value: st.models.m1?.[focusIdx], color: WIND },
                  { key: 'ensemble', value: st.models.ensemble?.[focusIdx], color: C.paper },
                ].map((point) =>
                  point.value == null ? null : (
                    <circle key={point.key} cx={cx(focusIdx)} cy={yTop(point.value)} r="2.8" fill={point.color} stroke={C.ink} strokeWidth="1">
                      <title>{`${point.key}: ${fmt(point.value)} µg/m³`}</title>
                    </circle>
                  ),
                )}
              </g>
            )}
            <rect x="0" y="0" width={iw} height={TOP_H} fill="transparent" className="cursor-crosshair" {...interactionProps} />
          </g>

          <text x={LEFT} y={BOTTOM_Y - 15} fontSize={11} fontWeight={500} fill={C.paper}>
            Diurnal sea-breeze & traffic wave D_s(t) = Y_s(t) − L_s(t)
          </text>
          <text x={LEFT + iw} y={BOTTOM_Y - 15} textAnchor="end" fontSize={10} fill={C.faint}>
            + trapping / traffic · − sea-breeze clearing
          </text>
          <g transform={`translate(${LEFT},${BOTTOM_Y})`}>
            {diurnalTicks.map((tick) => (
              <g key={tick}>
                <line x1="0" x2={iw} y1={yDiurnal(tick)} y2={yDiurnal(tick)} stroke={tick === 0 ? C.gridStrong : C.grid} />
                <text x="-10" y={yDiurnal(tick)} dy="0.32em" textAnchor="end" className="font-mono" fontSize={10} fill={C.faint}>
                  {tick > 0 ? '+' : ''}{tick}
                </text>
              </g>
            ))}
            {dayTicks.map((tick) => (
              <g key={tick.i}>
                <line x1={x(tick.i)} x2={x(tick.i)} y1="0" y2={BOTTOM_H} stroke={C.grid} />
                <text x={x(tick.i) + 4} y={BOTTOM_H + 18} className="font-mono" fontSize={10} fill={C.mute}>
                  {tick.label}
                </text>
              </g>
            ))}
            <g clipPath="url(#tm-diurnal-clip)">
              <path d={paths.diurnalArea} fill={TRAFFIC} fillOpacity="0.24" clipPath="url(#tm-positive-clip)" />
              <path d={paths.diurnalArea} fill={WIND} fillOpacity="0.2" clipPath="url(#tm-negative-clip)" />
              <path d={paths.diurnal} fill="none" stroke={C.paper} strokeOpacity="0.7" strokeWidth="1.25" />
            </g>
            {focusIdx != null && (
              <g pointerEvents="none">
                <line x1={cx(focusIdx)} x2={cx(focusIdx)} y1="0" y2={BOTTOM_H} stroke={C.paper} strokeOpacity="0.3" />
                {st.diurnal[focusIdx] != null && (
                  <circle
                    cx={cx(focusIdx)}
                    cy={yDiurnal(st.diurnal[focusIdx]!)}
                    r="3.2"
                    fill={st.diurnal[focusIdx]! >= 0 ? TRAFFIC : WIND}
                    stroke={C.ink}
                    strokeWidth="1"
                  />
                )}
              </g>
            )}
            <rect x="0" y="0" width={iw} height={BOTTOM_H} fill="transparent" className="cursor-crosshair" {...interactionProps} />
          </g>
        </svg>

        <div className="grid grid-cols-2 gap-3 border-t border-line pt-3 sm:grid-cols-5">
          <Readout label="Selected hour" value={`${dayLabel(fc.hours[readIdx])} · ${fc.hours[readIdx].slice(11, 16)} IST`} muted />
          <Readout label="L_s(t) synoptic" value={fmt(st.level[readIdx])} />
          <Readout label="D_s(t) diurnal" value={fmt(st.diurnal[readIdx])} signed />
          <Readout label="Raw CAMS → M1" value={`${fmt(st.cams[readIdx])} → ${fmt(st.models.m1?.[readIdx])}`} />
          <Readout label="Ensemble" value={fmt(st.models.ensemble?.[readIdx])} strong />
        </div>
        </div>
      </div>
    </section>
  )
}

function Readout({ label, value, signed = false, strong = false, muted = false }: { label: string; value: string; signed?: boolean; strong?: boolean; muted?: boolean }) {
  const numeric = Number(value)
  const shown = signed && value !== '—' && Number.isFinite(numeric) && numeric > 0 ? `+${value}` : value
  return (
    <div className="min-w-0">
      <div className="truncate font-mono text-[10px] text-faint">{label}</div>
      <div className={cx('mt-0.5 truncate font-cond text-[15px] tnum', muted ? 'font-sans text-[11px] text-mute' : strong ? 'font-semibold text-paper' : 'text-mute')}>
        {shown}
      </div>
    </div>
  )
}
