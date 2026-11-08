import { scaleLinear } from 'd3-scale'
import { area, curveStepAfter, line } from 'd3-shape'
import type { Forecasts } from '../../data/types'
import { pmColor } from '../../lib/aqi'
import { fmt } from '../../lib/series'
import { istParts } from '../../lib/stations'
import { C, EmptyNote, SectionHead, useWidth } from './shared'

const M = { top: 10, right: 14, bottom: 28, left: 40 }
const DENS_H = 70
const CDF_H = 150
const GAP = 14

type Marker = { key: string; label: string; hint: string; value: number; stroke: string; dash?: string; cap: 'dot' | 'square' | 'diamond' | 'none' }

export function MeanMedianSection({ fc, sid, hour, stationLabel }: { fc: Forecasts; sid: number; hour: number; stationLabel: string }) {
  const [ref, width] = useWidth<HTMLDivElement>(560)
  const st = fc.stations[String(sid)]
  const q21 = st.q21[hour]
  const dec = st.deciles[hour]
  const p = istParts(fc.hours[hour])
  const levels21 = fc.meta.quantiles21
  const levels9 = fc.meta.deciles

  const head = (
    <SectionHead
      eyebrow="Point forecast"
      title="Mean, not median"
      subtitle={`${stationLabel} · ${p.day} ${p.date}, ${p.time} IST. The forecast is a full distribution; the leaderboard needs one number per hour.`}
    />
  )

  if (!q21 || !dec || q21.some((v) => v == null) || dec.some((v) => v == null)) {
    return (
      <section>
        {head}
        <EmptyNote>No forecast distribution for this hour (it falls in a period without test rows). Move the time cursor to another hour.</EmptyNote>
      </section>
    )
  }
  const q = q21 as number[]
  const d = dec as number[]
  const median = q[levels21.indexOf(0.5)] ?? d[4]
  const tfm = st.models.timesfm_mean?.[hour] ?? null
  const chr = st.models.chronos_mean?.[hour] ?? null
  const ens = st.models.ensemble?.[hour] ?? null

  const markers: Marker[] = [
    { key: 'median', label: 'Median (q50)', hint: 'half the outcomes fall below', value: median, stroke: C.mute, dash: '3 3', cap: 'none' },
  ]
  if (tfm != null)
    markers.push({ key: 'tfm', label: 'TimesFM decile mean', hint: '0.15·(q10+q90) + 0.10·(q20…q80)', value: tfm, stroke: C.paper, cap: 'square' })
  if (chr != null)
    markers.push({ key: 'chr', label: 'Chronos-2 quantile mean', hint: 'trapezoidal ∫Q(p)dp over 21 levels', value: chr, stroke: C.paper, cap: 'diamond' })
  if (ens != null)
    markers.push({ key: 'ens', label: 'Ensemble forecast', hint: 'what we submit for this hour', value: ens, stroke: pmColor(ens), cap: 'dot' })

  const lowerTail = median - d[0]
  const upperTail = d[8] - median
  const skewed = upperTail > lowerTail * 1.08

  const iw = Math.max(200, width - M.left - M.right)
  const lo = Math.min(q[0], ...markers.map((m) => m.value))
  const hi = Math.max(q[q.length - 1], ...markers.map((m) => m.value))
  const x = scaleLinear().domain([lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04]).range([0, iw]).nice()

  // Piecewise-constant density implied by the 21 quantiles: Δp / Δx between neighbours.
  const dens = q.slice(0, -1).map((v, i) => {
    const dx = Math.max(1e-6, q[i + 1] - v)
    return { x0: v, x1: q[i + 1], h: (levels21[i + 1] - levels21[i]) / dx }
  })
  const hMax = Math.max(...dens.map((s) => s.h))
  const yd = scaleLinear().domain([0, hMax]).range([DENS_H, 6])
  const densPts = [...dens.map((s) => ({ x: s.x0, h: s.h })), { x: dens[dens.length - 1].x1, h: 0 }]
  const densPath = area<{ x: number; h: number }>()
    .curve(curveStepAfter)
    .x((s) => x(s.x))
    .y0(DENS_H)
    .y1((s) => yd(s.h))(densPts)

  const cdfTop = DENS_H + GAP
  const yc = scaleLinear().domain([0, 1]).range([cdfTop + CDF_H, cdfTop])
  const cdfPath = line<number>()
    .x((v) => x(v))
    .y((_, i) => yc(levels21[i]))(q)
  const totalH = M.top + DENS_H + GAP + CDF_H + M.bottom

  return (
    <section>
      {head}
      <div className="panel-solid grid gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_250px]">
        <div ref={ref} className="min-w-0">
          <svg width={width} height={totalH} role="img" aria-label="Forecast distribution with median and mean markers">
            <g transform={`translate(${M.left},${M.top})`}>
              {/* density */}
              <text x={-8} y={10} textAnchor="end" className="font-mono" fontSize={10} fill={C.faint}>
                dens.
              </text>
              <line x1={0} x2={iw} y1={DENS_H} y2={DENS_H} stroke={C.gridStrong} />
              <path d={densPath ?? ''} fill="rgba(236,231,220,0.14)" stroke="rgba(236,231,220,0.45)" strokeWidth={1} />

              {/* CDF grid */}
              {[0, 0.1, 0.5, 0.9, 1].map((pv) => (
                <g key={pv}>
                  <line x1={0} x2={iw} y1={yc(pv)} y2={yc(pv)} stroke={pv === 0.5 ? C.gridStrong : C.grid} />
                  <text x={-8} y={yc(pv)} dy="0.32em" textAnchor="end" className="font-mono" fontSize={11} fill={C.faint}>
                    {pv === 0 || pv === 1 ? pv : `q${Math.round(pv * 100)}`}
                  </text>
                </g>
              ))}
              {x.ticks(6).map((t) => (
                <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={cdfTop} y2={cdfTop + CDF_H} stroke={C.grid} />
                  <text x={x(t)} y={cdfTop + CDF_H + 18} textAnchor="middle" className="font-mono" fontSize={11} fill={C.faint}>
                    {t}
                  </text>
                </g>
              ))}
              <text x={iw} y={cdfTop + CDF_H + 18} dx={-2} textAnchor="end" className="font-mono" fontSize={10} fill={C.faint} dy={0}>
                µg/m³
              </text>

              {/* 80% range bracket */}
              <rect x={x(d[0])} width={x(d[8]) - x(d[0])} y={cdfTop + 1} height={CDF_H - 1} fill="rgba(236,231,220,0.03)" />

              {/* quantile curve */}
              <path d={cdfPath ?? ''} fill="none" stroke={C.mute} strokeWidth={1.3} />
              {q.map((v, i) => (
                <circle key={`q${i}`} cx={x(v)} cy={yc(levels21[i])} r={1.8} fill={C.mute} />
              ))}
              {d.map((v, i) => (
                <rect key={`d${i}`} x={x(v) - 3} y={yc(levels9[i]) - 3} width={6} height={6} fill={C.ink2} stroke={C.paper} strokeWidth={1.1} />
              ))}

              {/* markers */}
              {markers.map((m) => (
                <g key={m.key}>
                  <line
                    x1={x(m.value)}
                    x2={x(m.value)}
                    y1={0}
                    y2={cdfTop + CDF_H}
                    stroke={m.stroke}
                    strokeOpacity={m.key === 'median' ? 0.9 : 0.75}
                    strokeWidth={m.key === 'ens' ? 1.6 : 1}
                    strokeDasharray={m.dash}
                  />
                  <Cap kind={m.cap} x={x(m.value)} y={-2} color={m.stroke} />
                </g>
              ))}
            </g>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 pl-10 text-2xs text-faint">
            <span className="inline-flex items-center gap-1.5">
              <svg width="8" height="8" aria-hidden>
                <rect x="1" y="1" width="6" height="6" fill={C.ink2} stroke={C.paper} />
              </svg>
              9 deciles (TimesFM 3.0 slots)
            </span>
            <span className="inline-flex items-center gap-1.5">
              <svg width="8" height="8" aria-hidden>
                <circle cx="4" cy="4" r="2" fill={C.mute} />
              </svg>
              21 quantiles, q01–q99 (Chronos-2 slots)
            </span>
            <span>Top strip: density implied by the quantiles</span>
          </div>
        </div>

        <div className="flex flex-col">
          <ul className="space-y-2.5">
            {markers.map((m) => (
              <li key={m.key} className="flex items-start gap-3">
                <svg width="14" height="18" className="mt-0.5 shrink-0" aria-hidden>
                  <line x1="7" x2="7" y1="6" y2="18" stroke={m.stroke} strokeDasharray={m.dash} strokeWidth={1.3} />
                  <Cap kind={m.cap} x={7} y={4} color={m.stroke} />
                </svg>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[12px] text-paper">{m.label}</span>
                    <span className="font-mono text-[12px] tnum text-paper">{fmt(m.value)}</span>
                  </div>
                  <div className="truncate font-mono text-[10px] text-faint" title={m.hint}>
                    {m.hint}
                  </div>
                </div>
              </li>
            ))}
          </ul>
          <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-3">
            <div>
              <div className="text-2xs text-faint">Below median (q10 → q50)</div>
              <div className="font-mono text-[12px] tnum text-mute">−{fmt(lowerTail)}</div>
            </div>
            <div>
              <div className="text-2xs text-faint">Above median (q50 → q90)</div>
              <div className="font-mono text-[12px] tnum text-paper">+{fmt(upperTail)}</div>
            </div>
          </div>
          <p className="mt-4 text-[12.5px] leading-relaxed text-mute">
            RMSE squares every miss, so under-forecasting a pollution spike costs far more than small misses elsewhere. The single
            number that minimises squared error is the <span className="text-paper">mean</span> of the distribution, not the
            median.{' '}
            {skewed
              ? 'Here the upper tail is longer than the lower one, so the mean sits above the median; submitting the median would systematically fall short.'
              : 'At this hour the distribution is close to symmetric, so the mean and median nearly agree.'}
          </p>
        </div>
      </div>
    </section>
  )
}

function Cap({ kind, x, y, color }: { kind: Marker['cap']; x: number; y: number; color: string }) {
  if (kind === 'dot') return <circle cx={x} cy={y} r={4} fill={color} stroke={C.ink2} strokeWidth={1.5} />
  if (kind === 'square') return <rect x={x - 3} y={y - 3} width={6} height={6} fill={color} />
  if (kind === 'diamond') return <rect x={x - 3} y={y - 3} width={6} height={6} fill={color} transform={`rotate(45 ${x} ${y})`} />
  return null
}
