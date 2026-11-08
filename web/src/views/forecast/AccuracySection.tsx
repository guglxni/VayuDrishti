import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import type { Forecasts, LeaderRow } from '../../data/types'
import { cx } from '../../ui/primitives'
import { BASELINE_SHORT, C, SHORT_MODEL, SectionHead, seriesKey, useWidth } from './shared'

const ENSEMBLE_KEY = 'm7_stacked_ensemble'
const STARTER_KEY = 'b1_starter_rf_90d'
const X_DOMAIN: [number, number] = [8, 19]
const LABEL_W = 210
const VALUE_W = 108
const SPARK_W = 70
const ROW_H = 34

type Group = { title: string; rows: LeaderRow[] }

function groupRows(rows: LeaderRow[]): Group[] {
  const base = rows.filter((r) => r.key.startsWith('b'))
  const models = rows.filter((r) => /^m\d_/.test(r.key) && r.key !== ENSEMBLE_KEY)
  const ens = rows.filter((r) => r.key === ENSEMBLE_KEY)
  return [
    { title: 'Baselines', rows: base },
    { title: 'Individual models', rows: models },
    { title: 'Stacked ensemble', rows: ens },
  ].filter((g) => g.rows.length > 0)
}

function rowLabel(r: LeaderRow): { tag: string | null; label: string } {
  if (BASELINE_SHORT[r.key]) return { tag: null, label: BASELINE_SHORT[r.key] }
  const k = seriesKey(r.key)
  return { tag: k ? k.toUpperCase() : null, label: (k && SHORT_MODEL[k]) || r.name }
}

export function AccuracySection({ fc }: { fc: Forecasts }) {
  const [ref, width] = useWidth<HTMLDivElement>(900)
  const rows = fc.meta.leaderboard
  const groups = groupRows(rows)
  const showSpark = width >= 720
  const plotW = Math.max(160, width - LABEL_W - VALUE_W - (showSpark ? SPARK_W + 16 : 0) - 32)
  const x = scaleLinear().domain(X_DOMAIN).range([8, plotW - 18]).clamp(true)

  const ens = rows.find((r) => r.key === ENSEMBLE_KEY)
  const starter = rows.find((r) => r.key === STARTER_KEY)
  const gain = ens && starter ? 1 - ens.rmse24 / starter.rmse24 : null
  const precise24 = (fc.meta.ensemble_weights.gap24h as { oof_rmse_after_shrinkage?: number } | undefined)?.oof_rmse_after_shrinkage

  const sparkRows = rows.filter((r) => r.key !== 'b0_station_mean_90d').flatMap((r) => r.folds24)
  const sparkDom: [number, number] = [Math.min(...sparkRows), Math.max(...sparkRows)]

  return (
    <section>
      <SectionHead
        eyebrow="Accuracy"
        title="How far off the forecasts were, before the real test"
        subtitle="Out-of-fold RMSE over five rolling post-monsoon folds (Oct 12–21), µg/m³. Lower is better."
        right={
          ens && starter && gain != null ? (
            <div className="flex items-end gap-6">
              <div>
                <div className="text-2xs text-faint">Starter random forest → ensemble, 24 h ahead</div>
                <div className="mt-0.5 flex items-baseline gap-2 font-cond tnum">
                  <span className="text-[22px] text-mute">{starter.rmse24.toFixed(2)}</span>
                  <span className="text-faint">→</span>
                  <span className="text-[22px] font-semibold text-paper">{ens.rmse24.toFixed(2)}</span>
                  <span className="text-2xs text-mute">µg/m³</span>
                </div>
              </div>
              <div className="border-l border-line pl-6">
                <div className="text-2xs text-faint">Error reduction</div>
                <div className="mt-0.5 font-cond text-[22px] font-semibold tnum text-paper">
                  {Math.round(gain * 100)}%
                </div>
              </div>
            </div>
          ) : null
        }
      />

      <div ref={ref} className="panel-solid px-4 pb-4 pt-3">
        {/* Header / axis */}
        <div className="flex items-end gap-4 border-b border-line pb-2">
          <div style={{ width: LABEL_W }} className="shrink-0">
            <span className="flex items-center gap-4 text-2xs text-mute">
              <span className="flex items-center gap-1.5">
                <svg width="10" height="10" aria-hidden>
                  <circle cx="5" cy="5" r="4" fill={C.paper} />
                </svg>
                24 h ahead
              </span>
              <span className="flex items-center gap-1.5">
                <svg width="10" height="10" aria-hidden>
                  <circle cx="5" cy="5" r="3.6" fill="none" stroke={C.paper} strokeWidth={1.3} />
                </svg>
                72 h ahead
              </span>
            </span>
          </div>
          <svg width={plotW} height={18} className="shrink-0 overflow-visible">
            {x.ticks(6).map((t) => (
              <text key={t} x={x(t)} y={12} textAnchor="middle" className="font-mono" fontSize={11} fill={C.faint}>
                {t}
              </text>
            ))}
          </svg>
          <div style={{ width: VALUE_W }} className="shrink-0 text-right font-mono text-2xs text-faint">
            24 h · 72 h
          </div>
          {showSpark && (
            <div style={{ width: SPARK_W }} className="shrink-0 text-right font-mono text-2xs text-faint" title="RMSE in each of the five validation folds, 24 h ahead">
              5 folds
            </div>
          )}
        </div>

        {groups.map((g) => (
          <div key={g.title} className="mt-3">
            <div className="eyebrow mb-1">{g.title}</div>
            {g.rows.map((r) => {
              const isEns = r.key === ENSEMBLE_KEY
              const { tag, label } = rowLabel(r)
              const off24 = r.rmse24 > X_DOMAIN[1]
              const v72 = r.rmse72
              const off72 = v72 != null && v72 > X_DOMAIN[1]
              const ink = isEns ? C.paper : C.mute
              return (
                <div
                  key={r.key}
                  className={cx('-mx-2 flex items-center gap-4 rounded-lg px-2', isEns && 'bg-paper/[0.05]')}
                  style={{ height: ROW_H }}
                  title={r.name}
                >
                  <div style={{ width: LABEL_W }} className="flex min-w-0 shrink-0 items-baseline gap-2">
                    {tag && <span className="w-6 shrink-0 font-mono text-2xs text-faint">{tag}</span>}
                    <span className={cx('truncate text-[13px]', isEns ? 'font-semibold text-paper' : 'text-mute')}>
                      {label}
                    </span>
                  </div>
                  <svg width={plotW} height={ROW_H} className="shrink-0">
                    {x.ticks(6).map((t) => (
                      <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={ROW_H} stroke={C.grid} />
                    ))}
                    {ens && (
                      <line x1={x(ens.rmse24)} x2={x(ens.rmse24)} y1={0} y2={ROW_H} stroke="rgba(236,231,220,0.22)" strokeDasharray="2 3" />
                    )}
                    {v72 != null && (
                      <line
                        x1={x(r.rmse24)}
                        x2={x(v72)}
                        y1={ROW_H / 2}
                        y2={ROW_H / 2}
                        stroke={ink}
                        strokeOpacity={isEns ? 0.6 : 0.35}
                        strokeWidth={isEns ? 2 : 1.5}
                      />
                    )}
                    {v72 != null && (
                      <circle cx={x(v72)} cy={ROW_H / 2} r={isEns ? 4.5 : 4} fill={C.ink2} stroke={ink} strokeWidth={1.4} />
                    )}
                    <circle cx={x(r.rmse24)} cy={ROW_H / 2} r={isEns ? 5 : 4} fill={ink} />
                    {(off24 || off72) && (
                      <text x={plotW - 4} y={ROW_H / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={C.faint}>
                        off scale →
                      </text>
                    )}
                  </svg>
                  <div
                    style={{ width: VALUE_W }}
                    className={cx('shrink-0 text-right font-mono text-[12px] tnum', isEns ? 'text-paper' : 'text-mute')}
                  >
                    {r.rmse24.toFixed(2)}
                    <span className="text-faint"> · </span>
                    {v72 != null ? v72.toFixed(2) : '—'}
                  </div>
                  {showSpark && <Spark folds={r.folds24} dom={sparkDom} strong={isEns} />}
                </div>
              )
            })}
          </div>
        ))}

        <p className="mt-4 border-t border-line pt-3 text-2xs leading-relaxed text-faint">
          Each fold forecasts 48 hours after a reporting gap, using only data that ended before the forecast was issued.
          The dashed guide marks the ensemble's 24 h error
          {precise24 != null ? ` (${precise24.toFixed(4)} µg/m³ unrounded)` : ''}. Errors are scored against the raw,
          uncleaned sensor readings, exactly as the leaderboard does.
        </p>
      </div>
    </section>
  )
}

function Spark({ folds, dom, strong }: { folds: number[]; dom: [number, number]; strong: boolean }) {
  const w = SPARK_W
  const h = 20
  const own = Math.max(...folds) > dom[1] || Math.min(...folds) < dom[0]
  const d = own ? [Math.min(...folds), Math.max(...folds)] : dom
  const sx = scaleLinear().domain([0, folds.length - 1]).range([3, w - 3])
  const sy = scaleLinear().domain(d).range([h - 3, 3])
  const path = line<number>()
    .x((_, i) => sx(i))
    .y((v) => sy(v))(folds)
  return (
    <svg
      width={w}
      height={h}
      className="shrink-0"
      aria-label={`Fold RMSE: ${folds.join(', ')}`}
    >
      <title>{`Fold RMSE (24 h): ${folds.map((f) => f.toFixed(2)).join(' · ')}${own ? ' (own scale)' : ''}`}</title>
      <path
        d={path ?? ''}
        fill="none"
        stroke={strong ? C.paper : C.mute}
        strokeOpacity={strong ? 1 : 0.7}
        strokeWidth={strong ? 1.5 : 1.1}
        strokeDasharray={own ? '2 2' : undefined}
      />
      {folds.map((v, i) => (
        <circle key={i} cx={sx(i)} cy={sy(v)} r={1.4} fill={strong ? C.paper : C.mute} />
      ))}
    </svg>
  )
}
