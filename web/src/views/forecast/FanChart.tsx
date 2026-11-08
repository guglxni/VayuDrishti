import { useMemo, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react'
import { scaleLinear } from 'd3-scale'
import { area, line } from 'd3-shape'
import type { Block, Forecasts, Num } from '../../data/types'
import { AQI_BANDS, NAAQS_24H, WHO_24H, pmColor } from '../../lib/aqi'
import { BLOCK_LABEL, istParts } from '../../lib/stations'
import { fmt } from '../../lib/series'
import { cx } from '../../ui/primitives'
import { C, OVERLAY_KEYS, SHORT_MODEL, dayLabel, useWidth, type OverlayKey } from './shared'

const H = 380
const M = { top: 34, right: 16, bottom: 30, left: 46 }
const FOUNDATION = new Set<string>(['m7', 'm8', 'timesfm_mean', 'chronos_mean'])
const WIND = '#CFE3F2'

type Pt = { i: number; v: number }
type Band = { i: number; lo: number; hi: number }

function pts(arr: Num[] | undefined, a: number, b: number): (Pt | null)[] {
  const out: (Pt | null)[] = []
  if (!arr) return out
  for (let i = a; i <= b; i++) {
    const v = arr[i]
    out.push(v == null || !Number.isFinite(v) ? null : { i, v })
  }
  return out
}

function bandPts(rows: (Num[] | null)[], lo: number, hi: number, a: number, b: number): (Band | null)[] {
  const out: (Band | null)[] = []
  for (let i = a; i <= b; i++) {
    const r = rows[i]
    const l = r?.[lo]
    const h = r?.[hi]
    out.push(l == null || h == null ? null : { i, lo: l, hi: h })
  }
  return out
}

const BLOCK_FILL: Record<Block, string> = {
  validation: 'rgba(236,231,220,0.30)',
  public: 'rgba(236,231,220,0.55)',
  gap: 'rgba(236,231,220,0.0)',
  unscored: 'rgba(236,231,220,0.16)',
  private: C.paper,
}

export function FanChart({
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
  setHour: (h: number) => void
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>()
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const [overlays, setOverlays] = useState<Set<OverlayKey>>(() => new Set())
  const [hoverKey, setHoverKey] = useState<OverlayKey | null>(null)
  const [showCams, setShowCams] = useState(false)

  const st = fc.stations[String(sid)]
  const [a, b] = range
  const iw = Math.max(200, width - M.left - M.right)
  const ih = H - M.top - M.bottom

  const visibleOverlays = useMemo(() => {
    const s = new Set(overlays)
    if (hoverKey) s.add(hoverKey)
    return [...s]
  }, [overlays, hoverKey])

  const x = useMemo(() => scaleLinear().domain([a, b + 1]).range([0, iw]), [a, b, iw])

  const y = useMemo(() => {
    let max = NAAQS_24H + 15
    for (let i = a; i <= b; i++) {
      const q = st.q21[i]
      const d = st.deciles[i]
      for (const v of [q?.[19], d?.[8], st.obs[i], st.models.ensemble?.[i], showCams ? st.cams[i] : null]) if (v != null && v > max) max = v
      for (const k of visibleOverlays) {
        const v = st.models[k]?.[i]
        if (v != null && v > max) max = v
      }
    }
    return scaleLinear().domain([0, max * 1.06]).range([ih, 0]).nice()
  }, [a, b, st, ih, visibleOverlays, showCams])

  const cx0 = (i: number) => x(i + 0.5)

  const paths = useMemo(() => {
    const ln = line<Pt | null>()
      .defined((p) => p != null)
      .x((p) => cx0(p!.i))
      .y((p) => y(p!.v))
    const ar = area<Band | null>()
      .defined((p) => p != null)
      .x((p) => cx0(p!.i))
      .y0((p) => y(p!.lo))
      .y1((p) => y(p!.hi))
    const models: Record<string, string> = {}
    for (const k of OVERLAY_KEYS) models[k] = ln(pts(st.models[k], a, b)) ?? ''
    return {
      outer: ar(bandPts(st.q21, 1, 19, a, b)) ?? '',
      d1090: ar(bandPts(st.deciles, 0, 8, a, b)) ?? '',
      d3070: ar(bandPts(st.deciles, 2, 6, a, b)) ?? '',
      ens: ln(pts(st.models.ensemble, a, b)) ?? '',
      cams: ln(pts(st.cams, a, b)) ?? '',
      models,
    }
  }, [st, a, b, x, y])

  const obsPts = useMemo(() => pts(st.obs, a, b).filter((p): p is Pt => p != null), [st, a, b])

  // Time axis: day boundaries always; 6-hourly ticks when there is room.
  const span = b - a + 1
  const pxPerDay = (iw / span) * 24
  const dayStep = pxPerDay >= 54 ? 1 : Math.ceil(54 / pxPerDay)
  const showSixHour = pxPerDay >= 150
  const showNight = pxPerDay >= 40
  const ticks = useMemo(() => {
    const days: { i: number; label: string; major: boolean }[] = []
    const six: number[] = []
    const nights: [number, number][] = []
    for (let i = a; i <= b + 1; i++) {
      const iso = fc.hours[Math.min(i, fc.hours.length - 1)]
      const h = i < fc.hours.length ? Number(iso.slice(11, 13)) : 0
      if (h === 0) {
        const dayNum = Math.floor(i / 24)
        days.push({ i, label: i < fc.hours.length ? dayLabel(iso) : '', major: dayNum % dayStep === 0 })
      } else if (h % 6 === 0) six.push(i)
      if (h === 19) nights.push([i, Math.min(i + 11, b + 1)])
    }
    // Night that started before the window.
    const h0 = Number(fc.hours[a].slice(11, 13))
    if (h0 < 6) nights.unshift([a, a + (6 - h0)])
    return { days, six, nights }
  }, [a, b, fc.hours, dayStep])

  const yTicks = AQI_BANDS.map((bd) => bd.min).filter((v) => v <= y.domain()[1])
  const yMax = y.domain()[1]

  const blocks = useMemo(() => {
    const segs: { block: Block; s: number; e: number }[] = []
    for (let i = a; i <= b; i++) {
      const bl = fc.blocks[i]
      const last = segs[segs.length - 1]
      if (last && last.block === bl) last.e = i + 1
      else segs.push({ block: bl, s: i, e: i + 1 })
    }
    return segs
  }, [a, b, fc.blocks])

  const priv = fc.windows.find((w) => w.id === 'private')
  const val = fc.windows.find((w) => w.id === 'validation')
  const cutoffIdx = fc.hours.findIndex((h) => h.slice(0, 16) === fc.meta.cutoff_t0.slice(0, 16))
  const clipX = (v: number) => Math.max(0, Math.min(iw, v))
  const inRange = (s: number, e: number) => e > a && s <= b
  const privateBadgeW = Math.min(278, Math.max(160, iw - 8))
  const privateBadgeX = priv
    ? Math.max(
        privateBadgeW / 2 + 4,
        Math.min(iw - privateBadgeW / 2 - 4, (clipX(x(priv.start)) + clipX(x(priv.end + 1))) / 2),
      )
    : 0

  function idxFromEvent(e: MouseEvent<SVGRectElement>) {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    return Math.max(a, Math.min(b, Math.floor(x.invert(px))))
  }

  function onKey(e: KeyboardEvent<SVGSVGElement>) {
    const step = e.shiftKey ? 24 : 1
    if (e.key === 'ArrowRight') setHour(hour + step)
    else if (e.key === 'ArrowLeft') setHour(hour - step)
    else return
    e.preventDefault()
  }

  const cursorIn = hour >= a && hour <= b
  const ensAtCursor = st.models.ensemble?.[hour] ?? null
  const hi = hoverIdx
  const tip = hi == null ? null : buildTip(fc, st, hi, hoverKey, showCams)
  const tipLeft = hi == null ? 0 : M.left + cx0(hi)
  const tipFlip = tipLeft > width - 230

  function toggle(k: OverlayKey) {
    setOverlays((prev) => {
      const n = new Set(prev)
      if (n.has(k)) n.delete(k)
      else n.add(k)
      return n
    })
  }

  return (
    <div>
      <div ref={wrapRef} className="relative select-none">
        <svg
          width={width}
          height={H}
          role="img"
          aria-label="Hourly PM2.5 forecast with uncertainty bands. Use arrow keys to move the time cursor."
          tabIndex={0}
          onKeyDown={onKey}
          className="block outline-none focus-visible:outline-2 focus-visible:outline-paper/40"
        >
          <defs>
            <pattern id="fc-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(236,231,220,0.07)" strokeWidth="2" />
            </pattern>
            <clipPath id="fc-clip">
              <rect x={0} y={-2} width={iw} height={ih + 4} />
            </clipPath>
          </defs>
          <g transform={`translate(${M.left},${M.top})`}>
            {/* Night shading */}
            {showNight &&
              ticks.nights.map(([s, e]) =>
                inRange(s, e) ? (
                  <rect
                    key={`n${s}`}
                    x={clipX(x(s))}
                    width={Math.max(0, clipX(x(e)) - clipX(x(s)))}
                    y={0}
                    height={ih}
                    fill="rgba(236,231,220,0.018)"
                  />
                ) : null,
              )}

            {/* Validation period */}
            {val && inRange(val.start, val.end + 1) && (
              <rect
                x={clipX(x(val.start))}
                width={clipX(x(val.end + 1)) - clipX(x(val.start))}
                y={0}
                height={ih}
                fill="rgba(236,231,220,0.022)"
              />
            )}

            {/* Gap with no test rows */}
            {blocks
              .filter((s) => s.block === 'gap')
              .map((s) => (
                <g key={`g${s.s}`}>
                  <rect x={x(s.s)} width={x(s.e) - x(s.s)} y={0} height={ih} fill="url(#fc-hatch)" />
                  {x(s.e) - x(s.s) > 70 && (
                    <text
                      x={(x(s.s) + x(s.e)) / 2}
                      y={ih / 2}
                      textAnchor="middle"
                      className="font-mono"
                      fontSize={10}
                      fill={C.faint}
                    >
                      no test rows
                    </text>
                  )}
                </g>
              ))}

            {/* Scored window */}
            {priv && inRange(priv.start, priv.end + 1) && (
              <g>
                <rect
                  x={clipX(x(priv.start))}
                  width={clipX(x(priv.end + 1)) - clipX(x(priv.start))}
                  y={0}
                  height={ih}
                  fill="rgba(236,231,220,0.045)"
                />
                {[priv.start, priv.end + 1].map((i) =>
                  i >= a && i <= b + 1 ? (
                    <line key={i} x1={x(i)} x2={x(i)} y1={0} y2={ih} stroke={C.gridStrong} strokeDasharray="2 3" />
                  ) : null,
                )}
              </g>
            )}

            {/* Horizontal grid + AQI strip */}
            {yTicks.map((v) => (
              <g key={v}>
                <line x1={0} x2={iw} y1={y(v)} y2={y(v)} stroke={v === 0 ? C.gridStrong : C.grid} />
                <text x={-12} y={y(v)} dy="0.32em" textAnchor="end" className="font-mono" fontSize={11} fill={C.faint}>
                  {v}
                </text>
              </g>
            ))}
            {AQI_BANDS.map((bd) => {
              const top = Math.min(bd.max, yMax)
              if (bd.min >= yMax) return null
              return (
                <rect key={bd.label} x={-6} width={3} y={y(top)} height={y(bd.min) - y(top)} fill={bd.color}>
                  <title>{`${bd.label}: ${bd.min}–${Number.isFinite(bd.max) ? bd.max : '+'} µg/m³`}</title>
                </rect>
              )
            })}

            {/* Day gridlines + time labels */}
            {ticks.days.map((d) => (
              <g key={`d${d.i}`}>
                <line x1={x(d.i)} x2={x(d.i)} y1={0} y2={ih} stroke={d.major ? C.grid : 'rgba(236,231,220,0.035)'} />
                {d.major && d.label && x(d.i) < iw - 20 && (
                  <text x={x(d.i) + 4} y={ih + 18} className="font-mono" fontSize={11} fill={C.mute}>
                    {d.label}
                  </text>
                )}
              </g>
            ))}
            {showSixHour &&
              ticks.six.map((i) => (
                <g key={`s${i}`}>
                  <line x1={x(i)} x2={x(i)} y1={ih} y2={ih + 4} stroke={C.gridStrong} />
                  <text x={x(i)} y={ih + 18} textAnchor="middle" className="font-mono" fontSize={10} fill={C.faint}>
                    {String(Number(fc.hours[i].slice(11, 13))).padStart(2, '0')}
                  </text>
                </g>
              ))}

            {/* Reference lines */}
            <line x1={0} x2={iw} y1={y(NAAQS_24H)} y2={y(NAAQS_24H)} stroke="rgba(236,231,220,0.42)" strokeDasharray="5 4" />
            <text x={iw - 4} y={y(NAAQS_24H) - 6} textAnchor="end" fontSize={11} fill={C.mute}>
              National 24-h standard · 60
            </text>
            <line x1={0} x2={iw} y1={y(WHO_24H)} y2={y(WHO_24H)} stroke="rgba(236,231,220,0.2)" strokeDasharray="1 4" />
            <text x={iw - 4} y={y(WHO_24H) - 6} textAnchor="end" fontSize={11} fill={C.faint}>
              WHO guideline · 15
            </text>

            <g clipPath="url(#fc-clip)">
              {/* Uncertainty */}
              <path d={paths.outer} fill="rgba(236,231,220,0.05)" />
              <path d={paths.d1090} fill="rgba(236,231,220,0.09)" />
              <path d={paths.d3070} fill="rgba(236,231,220,0.12)" />

              {/* Model overlays */}
              {visibleOverlays.map((k) => {
                const isHi = hoverKey === k
                const dim = hoverKey != null && !isHi
                return (
                  <path
                    key={k}
                    d={paths.models[k]}
                    fill="none"
                    stroke={isHi ? C.paper : C.mute}
                    strokeOpacity={isHi ? 0.95 : dim ? 0.18 : 0.55}
                    strokeWidth={isHi ? 1.5 : 1}
                    strokeDasharray={FOUNDATION.has(k) ? '4 3' : undefined}
                  />
                )
              })}

              {/* Raw uncalibrated CAMS */}
              {showCams && (
                <path d={paths.cams} fill="none" stroke={WIND} strokeOpacity={0.75} strokeWidth={1.25} strokeDasharray="5 4" />
              )}

              {/* Ensemble */}
              <path d={paths.ens} fill="none" stroke={C.ink} strokeWidth={4} strokeOpacity={0.6} />
              <path d={paths.ens} fill="none" stroke={C.paper} strokeWidth={1.75} strokeLinejoin="round" />

              {/* Observations */}
              {obsPts.map((p) => (
                <circle key={p.i} cx={cx0(p.i)} cy={y(p.v)} r={span > 300 ? 1.3 : 2} fill={C.ink} stroke={C.paper} strokeOpacity={0.85} strokeWidth={1} />
              ))}
            </g>

            {/* Forecast issue time */}
            {cutoffIdx >= a && cutoffIdx <= b && (
              <g>
                <line x1={x(cutoffIdx)} x2={x(cutoffIdx)} y1={-8} y2={ih} stroke={C.gridStrong} />
                <text x={x(cutoffIdx) + 5} y={-1} fontSize={11} fill={C.mute}>
                  Forecast issued 22 Oct, 00:00
                </text>
              </g>
            )}

            {/* Period labels */}
            {val && inRange(val.start, val.end + 1) && clipX(x(val.end + 1)) - clipX(x(val.start)) > 150 && (
              <text x={clipX(x(val.start)) + 6} y={14} fontSize={11} fill={C.mute}>
                Validation · out-of-fold forecast vs. observed
              </text>
            )}
            {priv && inRange(priv.start, priv.end + 1) && (
              <g pointerEvents="none" aria-label="Scored Private LB Window: Nov 11–12, 2026, 48 hours, 192 rows">
                <rect
                  x={privateBadgeX - privateBadgeW / 2}
                  y={3}
                  width={privateBadgeW}
                  height={42}
                  rx={6}
                  fill={C.paper}
                  stroke={C.ink}
                  strokeWidth={1}
                />
                <text x={privateBadgeX} y={18} textAnchor="middle" fontSize={9} fontWeight={700} letterSpacing={0.65} fill={C.ink}>
                  {privateBadgeW >= 250 ? 'SCORED PRIVATE LB WINDOW' : 'PRIVATE LB · 192 ROWS'}
                </text>
                <text x={privateBadgeX} y={34} textAnchor="middle" fontSize={11} fontWeight={600} fill={C.ink}>
                  {privateBadgeW >= 250 ? 'Nov 11–12, 2026 · 48h · 192 rows' : 'Nov 11–12, 2026 · 48h'}
                </text>
              </g>
            )}

            {/* Block ribbon */}
            {blocks.map((s) => (
              <rect
                key={`b${s.s}`}
                x={x(s.s)}
                width={Math.max(0, x(s.e) - x(s.s) - 1)}
                y={-M.top + 4}
                height={3}
                rx={1.5}
                fill={s.block === 'gap' ? 'url(#fc-hatch)' : BLOCK_FILL[s.block]}
              >
                <title>{BLOCK_LABEL[s.block]}</title>
              </rect>
            ))}

            {/* Cursor */}
            {cursorIn && (
              <g pointerEvents="none">
                <line x1={cx0(hour)} x2={cx0(hour)} y1={0} y2={ih} stroke={C.paper} strokeOpacity={0.55} />
                {ensAtCursor != null && (
                  <circle cx={cx0(hour)} cy={y(ensAtCursor)} r={4.5} fill={pmColor(ensAtCursor)} stroke={C.ink} strokeWidth={2} />
                )}
              </g>
            )}

            {/* Hover crosshair */}
            {hi != null && (
              <g pointerEvents="none">
                <line x1={cx0(hi)} x2={cx0(hi)} y1={0} y2={ih} stroke={C.paper} strokeOpacity={0.22} />
                {st.models.ensemble?.[hi] != null && (
                  <circle cx={cx0(hi)} cy={y(st.models.ensemble[hi]!)} r={3} fill={C.paper} />
                )}
                {st.obs[hi] != null && <circle cx={cx0(hi)} cy={y(st.obs[hi]!)} r={3.2} fill="none" stroke={C.paper} strokeWidth={1.4} />}
                {showCams && st.cams[hi] != null && (
                  <circle cx={cx0(hi)} cy={y(st.cams[hi]!)} r={3} fill={C.ink} stroke={WIND} strokeWidth={1.3} />
                )}
                {hoverKey && st.models[hoverKey]?.[hi] != null && (
                  <circle cx={cx0(hi)} cy={y(st.models[hoverKey][hi]!)} r={2.5} fill={C.mute} />
                )}
              </g>
            )}

            <rect
              x={0}
              y={0}
              width={iw}
              height={ih}
              fill="transparent"
              style={{ cursor: 'crosshair' }}
              onPointerMove={(e) => setHoverIdx(idxFromEvent(e))}
              onPointerLeave={() => setHoverIdx(null)}
              onClick={(e) => setHour(idxFromEvent(e))}
            />
          </g>
        </svg>

        {tip && (
          <div
            className="pointer-events-none absolute z-10 w-[214px] rounded-[10px] border border-line-strong bg-ink-2/95 px-3 py-2.5 shadow-[0_12px_32px_-12px_rgba(0,0,0,0.8)] backdrop-blur"
            style={{ left: tipFlip ? tipLeft - 226 : tipLeft + 12, top: M.top + 8 }}
          >
            {tip}
          </div>
        )}
      </div>

      {/* Legend + overlays */}
      <div className="mt-4 flex flex-wrap items-start justify-between gap-x-8 gap-y-4 border-t border-line pt-4">
        <Legend />
        <div className="min-w-0 flex-1 basis-[420px]">
          <div className="mb-2 flex items-baseline justify-between gap-3">
            <span className="eyebrow">Overlay individual models</span>
            {overlays.size > 0 && (
              <button className="text-2xs text-mute hover:text-paper" onClick={() => setOverlays(new Set())}>
                Clear
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <button
              aria-pressed={showCams}
              title="Raw CAMS · uncalibrated Open-Meteo prior"
              onClick={() => setShowCams((value) => !value)}
              className={cx(
                'inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-2xs transition-colors',
                showCams ? 'border-line-strong bg-paper/8 text-paper' : 'border-line text-mute hover:text-paper',
              )}
            >
              <svg width="16" height="6" aria-hidden>
                <line x1="0" x2="16" y1="3" y2="3" stroke={WIND} strokeWidth={1.4} strokeDasharray="5 4" />
              </svg>
              Raw CAMS
            </button>
            {OVERLAY_KEYS.map((k) => {
              const on = overlays.has(k)
              return (
                <button
                  key={k}
                  aria-pressed={on}
                  title={fc.meta.model_names[k] ?? k}
                  onClick={() => toggle(k)}
                  onPointerEnter={() => setHoverKey(k)}
                  onPointerLeave={() => setHoverKey(null)}
                  onFocus={() => setHoverKey(k)}
                  onBlur={() => setHoverKey(null)}
                  className={cx(
                    'inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-2xs transition-colors',
                    on ? 'border-line-strong bg-paper/8 text-paper' : 'border-line text-mute hover:text-paper',
                  )}
                >
                  <svg width="16" height="6" aria-hidden>
                    <line
                      x1="0"
                      x2="16"
                      y1="3"
                      y2="3"
                      stroke={on ? C.paper : C.mute}
                      strokeWidth={1.4}
                      strokeDasharray={FOUNDATION.has(k) ? '4 3' : undefined}
                    />
                  </svg>
                  {k.startsWith('m') && k.length === 2 && <span className="font-mono text-faint">{k.toUpperCase()}</span>}
                  {SHORT_MODEL[k]}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

function Legend() {
  const sw = (fill: string) => <span className="inline-block h-2.5 w-4 rounded-[2px]" style={{ background: fill }} />
  return (
    <div className="grid grid-cols-[auto_auto] gap-x-6 gap-y-1.5 text-2xs text-mute">
      <span className="flex items-center gap-2">
        <svg width="16" height="6" aria-hidden>
          <line x1="0" x2="16" y1="3" y2="3" stroke={C.paper} strokeWidth={1.75} />
        </svg>
        Ensemble forecast (mean)
      </span>
      <span className="flex items-center gap-2">
        <svg width="16" height="8" aria-hidden>
          <circle cx="8" cy="4" r="2.5" fill={C.ink} stroke={C.paper} />
        </svg>
        Observed (validation)
      </span>
      <span className="flex items-center gap-2">{sw('rgba(236,231,220,0.26)')}Middle 40% (q30–q70)</span>
      <span className="flex items-center gap-2">
        <svg width="16" height="6" aria-hidden>
          <line x1="0" x2="16" y1="3" y2="3" stroke="rgba(236,231,220,0.5)" strokeDasharray="5 4" />
        </svg>
        Safe limit, 60 µg/m³
      </span>
      <span className="flex items-center gap-2">{sw('rgba(236,231,220,0.14)')}80% range (q10–q90)</span>
      <span className="flex items-center gap-2">{sw('rgba(236,231,220,0.06)')}90% range (q5–q95)</span>
    </div>
  )
}

function buildTip(fc: Forecasts, st: Forecasts['stations'][string], i: number, hoverKey: OverlayKey | null, showCams: boolean) {
  const p = istParts(fc.hours[i])
  const ens = st.models.ensemble?.[i] ?? null
  const d = st.deciles[i]
  const obs = st.obs[i]
  const lead = fc.horizon[i]
  return (
    <div className="text-2xs">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-paper">
          {p.day} {p.date} · {p.time}
        </span>
        {lead > 0 && <span className="font-mono text-faint">+{lead}h</span>}
      </div>
      <div className="mt-0.5 text-faint">{BLOCK_LABEL[fc.blocks[i]]}</div>
      {ens == null ? (
        <div className="mt-2 text-mute">No forecast for this hour.</div>
      ) : (
        <div className="mt-2 space-y-1">
          <Row label="Ensemble">
            <span className="font-cond text-[15px] font-semibold tnum" style={{ color: pmColor(ens) }}>
              {fmt(ens)}
            </span>
          </Row>
          {d && <Row label="80% range">{`${fmt(d[0])} – ${fmt(d[8])}`}</Row>}
          {obs != null && <Row label="Observed">{fmt(obs)}</Row>}
          {showCams && st.cams[i] != null && <Row label="Raw CAMS">{fmt(st.cams[i])}</Row>}
          {hoverKey && st.models[hoverKey]?.[i] != null && <Row label={SHORT_MODEL[hoverKey]}>{fmt(st.models[hoverKey][i])}</Row>}
        </div>
      )}
      <div className="mt-2 border-t border-line pt-1.5 text-faint">Click to move the time cursor here</div>
    </div>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-mute">{label}</span>
      <span className="font-mono tnum text-paper">{children}</span>
    </div>
  )
}
