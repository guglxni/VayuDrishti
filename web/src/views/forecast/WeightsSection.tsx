import { useState } from 'react'
import type { Forecasts } from '../../data/types'
import { TARGET_IDS, TARGET_META } from '../../lib/stations'
import { cx } from '../../ui/primitives'
import { SHORT_MODEL, SectionHead, gapWeights, seriesKey, type GapWeights } from './shared'

/** Neutral greys, light → dark, cycled so neighbouring segments stay distinct. */
const GREYS = ['#D9D3C7', '#8C96A0', '#C4BEB2', '#7A8590', '#B0AA9E', '#97A1AB', '#CFC9BD', '#B9C1C9']

const GAPS: { key: 'gap24h' | 'gap72h'; label: string; hint: string }[] = [
  { key: 'gap24h', label: '24 h ahead', hint: 'forecast issued one day before' },
  { key: 'gap72h', label: '72 h ahead', hint: 'forecast issued three days before' },
]

export function WeightsSection({ fc, stationId }: { fc: Forecasts; stationId: number }) {
  const [hover, setHover] = useState<string | null>(null)
  const sets = GAPS.map((g) => ({ ...g, w: gapWeights(fc.meta, g.key) })).filter(
    (g): g is (typeof GAPS)[number] & { w: GapWeights } => g.w != null,
  )
  if (sets.length === 0) return null
  const keys = Object.keys(sets[0].w.weights)

  return (
    <section>
      <SectionHead
        eyebrow="Ensemble"
        title="How the eight models are weighted"
        subtitle="Weights were fitted on the out-of-fold forecasts to minimise RMSE. A small floor keeps every model in the blend; most weight goes to the models that generalised best."
      />

      <div className="panel-solid p-5">
        <div className="space-y-5">
          {sets.map((g) => (
            <div key={g.key}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <span className="text-[13px] text-paper">
                  {g.label} <span className="text-mute">· {g.hint}</span>
                </span>
                <span className="font-mono text-2xs text-faint tnum">
                  RMSE {g.w.oof_rmse_after_shrinkage.toFixed(4)}
                </span>
              </div>
              <div className="flex h-9 w-full overflow-hidden rounded-[8px]" onPointerLeave={() => setHover(null)}>
                {keys.map((k, i) => {
                  const v = g.w.weights[k] ?? 0
                  const sk = seriesKey(k) ?? k
                  const dim = hover != null && hover !== k
                  const big = v >= 0.12
                  return (
                    <div
                      key={k}
                      onPointerEnter={() => setHover(k)}
                      title={`${fc.meta.model_names[sk] ?? k}: ${(v * 100).toFixed(1)}%`}
                      className="relative flex min-w-0 items-center overflow-hidden border-r border-ink-2 px-2 transition-opacity last:border-r-0"
                      style={{ width: `${v * 100}%`, background: GREYS[i % GREYS.length], opacity: dim ? 0.3 : 1 }}
                    >
                      {big && (
                        <span className="truncate text-2xs font-medium text-ink">
                          {SHORT_MODEL[sk]} <span className="font-mono tnum opacity-75">{Math.round(v * 100)}%</span>
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Weight table */}
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-2xs text-faint">
                <th className="pb-2 font-normal">Model</th>
                {sets.map((g) => (
                  <th key={g.key} className="pb-2 text-right font-normal">
                    {g.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {keys.map((k, i) => {
                const sk = seriesKey(k) ?? k
                return (
                  <tr
                    key={k}
                    onPointerEnter={() => setHover(k)}
                    onPointerLeave={() => setHover(null)}
                    className={cx('border-t border-line', hover === k && 'bg-paper/[0.04]')}
                  >
                    <td className="py-1.5 pr-3">
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: GREYS[i % GREYS.length] }} />
                        <span className="w-6 font-mono text-2xs text-faint">{sk.toUpperCase()}</span>
                        <span className="text-mute" title={fc.meta.model_names[sk]}>
                          {fc.meta.model_names[sk] ?? k}
                        </span>
                      </span>
                    </td>
                    {sets.map((g) => {
                      const v = g.w.weights[k] ?? 0
                      return (
                        <td key={g.key} className="py-1.5 text-right">
                          <span className="inline-flex items-center justify-end gap-2">
                            <span className="h-1 rounded-full bg-paper/50" style={{ width: Math.max(2, v * 90) }} />
                            <span className={cx('w-11 font-mono tnum', v >= 0.12 ? 'text-paper' : 'text-faint')}>
                              {(v * 100).toFixed(1)}%
                            </span>
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Shrinkage */}
        <div className="mt-6 border-t border-line pt-5">
          <div className="text-[13px] font-medium text-paper">Per-station calibration</div>
          <p className="mt-1 text-2xs leading-relaxed text-mute">
            After blending, each station's forecast is pulled toward a physics prior (55% CAMS MOS prior, 45% physics GAM)
            and nudged by a bias shift:{' '}
            <span className="font-mono text-paper/85">final = α·blend + (1 − α)·prior + shift</span>. An α below 1 damps
            over-confident swings.
          </p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left text-2xs text-faint">
                  <th className="pb-2 font-normal">Station</th>
                  {sets.map((g) => (
                    <th key={g.key} colSpan={2} className="pb-2 text-right font-normal">
                      {g.label}: α · shift
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {TARGET_IDS.map((id) => (
                  <tr key={id} className={cx('border-t border-line', id === stationId && 'bg-paper/[0.04]')}>
                    <td className={cx('py-1.5', id === stationId ? 'text-paper' : 'text-mute')}>{TARGET_META[id].name}</td>
                    {sets.map((g) => {
                      const a = g.w.station_alphas?.[String(id)]
                      const s = g.w.station_shifts?.[String(id)]
                      return [
                        <td key={`${g.key}a`} className={cx('py-1.5 text-right font-mono tnum', a != null && a < 1 ? 'text-paper' : 'text-mute')}>
                          {a != null ? a.toFixed(2) : '—'}
                        </td>,
                        <td key={`${g.key}s`} className="py-1.5 pl-3 text-right font-mono tnum text-mute">
                          {s != null ? `${s >= 0 ? '+' : '−'}${Math.abs(s).toFixed(2)}` : '—'}
                        </td>,
                      ]
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            {sets.map((g) => (
              <div key={g.key} className="text-2xs text-mute">
                <div className="text-faint">{g.label}: RMSE before → after calibration</div>
                <div className="mt-0.5 font-mono tnum text-paper">
                  {g.w.oof_rmse_before_shrinkage != null ? g.w.oof_rmse_before_shrinkage.toFixed(4) : '—'}
                  <span className="text-faint"> → </span>
                  {g.w.oof_rmse_after_shrinkage.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
