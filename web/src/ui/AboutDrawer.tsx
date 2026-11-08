import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ExternalLink, Search, X } from 'lucide-react'
import { useData, useTwin } from '../state/store'
import { cx, Segmented } from './primitives'

const MODEL_ROWS: { key: string; weightKey: string; plain: string }[] = [
  { key: 'm1_calibrated_cams_prior', weightKey: 'm1_calibrated_cams_prior', plain: 'Daily rhythm + calibrated CAMS chemistry forecast' },
  { key: 'm2_huber_ridge_gam', weightKey: 'm2_huber_ridge_gam', plain: 'Robust regression on weather and physics features' },
  { key: 'm3_lightgbm_residual', weightKey: 'm3_lightgbm_residual', plain: 'LightGBM trees correcting the baseline' },
  { key: 'm4_catboost_residual', weightKey: 'm4_catboost_residual', plain: 'CatBoost trees correcting the baseline' },
  { key: 'm5_xgboost_residual', weightKey: 'm5_xgboost_residual', plain: 'XGBoost trees correcting the baseline' },
  { key: 'm6_wind_st_gat', weightKey: 'm6_wind_st_gat', plain: 'Graph network passing pollution along the wind' },
  { key: 'm7_timesfm3_compact', weightKey: 'm7_timesfm3_compact', plain: 'Google TimesFM 3.0 foundation model' },
  { key: 'm8_chronos2_covariate', weightKey: 'm8_chronos2_covariate', plain: 'Amazon Chronos-2 foundation model with weather inputs' },
]

const STEPS = [
  {
    title: 'Clean the sensor record',
    body: 'Readings flagged above 1,000 or over 450 µg/m³ are masked, then a 7-hour Hampel filter removes isolated spikes. Models train on the cleaned series; accuracy is scored against the raw readings, exactly as Zindi does.',
  },
  {
    title: 'Build features without peeking',
    body: 'Every feature for a forecast made at time T uses only hours that ended by T: station history, Open-Meteo weather, CAMS chemistry, airport winds (no visibility or weather codes), sea-breeze and Deonar-landfill wind alignment.',
  },
  {
    title: 'Run eight different forecasters',
    body: 'Physics-style regressions, three gradient-boosted tree models, a wind-guided graph network and two pretrained time-series foundation models each forecast all 2,592 test hours.',
  },
  {
    title: 'Blend them on held-out folds',
    body: 'Weights are fitted on five rolling post-monsoon folds (Oct 12–21) at 24 h and 72 h reporting gaps, then each station gets a variance shrinkage that tames over-confident swings.',
  },
  {
    title: 'Report the expected value and its range',
    body: 'Because the score is RMSE, point forecasts are the mean of the predicted distribution (integrated over all quantiles), not its median. The 80% range shown on the map comes from split-conformal calibration on the folds.',
  },
]

export function AboutDrawer() {
  const open = useTwin((s) => s.aboutOpen)
  const setOpen = useTwin((s) => s.setAboutOpen)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="absolute inset-0 z-40 bg-ink/60"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
          />
          <motion.aside
            role="dialog"
            aria-modal="true"
            aria-label="About the model"
            className="absolute bottom-0 right-0 top-0 z-50 flex w-full max-w-[560px] flex-col border-l border-line bg-ink-2"
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ type: 'tween', duration: 0.22 }}
          >
            <DrawerBody onClose={() => setOpen(false)} />
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

function DrawerBody({ onClose }: { onClose: () => void }) {
  const data = useData()
  const [tab, setTab] = useState<'model' | 'data'>('model')
  const meta = data.forecasts.meta
  const w24 = meta.ensemble_weights.gap24h
  const w72 = meta.ensemble_weights.gap72h
  const lb = new Map(meta.leaderboard.map((r) => [r.key, r]))
  const rf = lb.get('b1_starter_rf_90d')

  return (
    <>
      <div className="flex items-start justify-between border-b border-line px-6 py-5">
        <div>
          <div className="eyebrow">About the model</div>
          <h2 className="mt-1 text-[19px] font-semibold tracking-tight">How VayuDrishti forecasts Mumbai's air</h2>
          <p className="mt-1 max-w-[440px] text-[13px] text-mute">
            Built for the Zindi × Google Mumbai Air Quality Data Challenge. Forecasts cover 22 Oct – 20 Nov 2026 for four CPCB stations, made
            only from data available by 22 Oct 00:00 IST.
          </p>
        </div>
        <button onClick={onClose} className="rounded-md p-1.5 text-mute hover:bg-paper/8 hover:text-paper" aria-label="Close">
          <X size={18} />
        </button>
      </div>
      <div className="px-6 pt-4">
        <Segmented
          size="sm"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'model', label: 'Model and accuracy' },
            { value: 'data', label: `Data sources (${data.catalog.meta.count})` },
          ]}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-10 pt-5">
        {tab === 'model' ? (
          <div className="space-y-8">
            <section>
              <div className="grid grid-cols-2 gap-4">
                <Metric value={w24.oof_rmse_after_shrinkage} label="Typical error, 24 h reporting gap" />
                <Metric value={w72.oof_rmse_after_shrinkage} label="Typical error, 72 h reporting gap" />
              </div>
              <p className="mt-3 text-xs text-mute">
                Out-of-fold RMSE in µg/m³ across five rolling folds, Oct 12–21 2026.
                {rf && (
                  <>
                    {' '}
                    The challenge's starter random forest scores {rf.rmse24.toFixed(2)} on the same folds, so the blend cuts error by{' '}
                    {Math.round((1 - w24.oof_rmse_after_shrinkage / rf.rmse24) * 100)}%.
                  </>
                )}
              </p>
            </section>

            <section>
              <h3 className="text-[15px] font-medium">Pipeline</h3>
              <ol className="mt-3 space-y-4">
                {STEPS.map((s, i) => (
                  <li key={s.title} className="grid grid-cols-[28px_1fr] gap-2">
                    <span className="font-mono text-xs text-faint tnum">{String(i + 1).padStart(2, '0')}</span>
                    <div>
                      <div className="text-[13px] font-medium">{s.title}</div>
                      <p className="mt-0.5 text-[13px] leading-relaxed text-mute">{s.body}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            <section>
              <h3 className="text-[15px] font-medium">The eight models</h3>
              <p className="mt-1 text-xs text-mute">Error is out-of-fold RMSE at the 24 h gap. Weight is each model's share of the final blend.</p>
              <table className="mt-3 w-full text-left text-[13px]">
                <thead>
                  <tr className="text-2xs text-faint">
                    <th className="pb-2 font-normal">Model</th>
                    <th className="pb-2 text-right font-normal">Error</th>
                    <th className="pb-2 pl-3 text-right font-normal">Weight 24 h</th>
                    <th className="pb-2 pl-3 text-right font-normal">72 h</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_ROWS.map((m, i) => {
                    const row = lb.get(m.key)
                    const a = w24.weights[m.weightKey] ?? 0
                    const b = w72.weights[m.weightKey] ?? 0
                    return (
                      <tr key={m.key} className="border-t border-line align-top">
                        <td className="py-2 pr-2">
                          <div className="text-paper">
                            <span className="mr-1.5 font-mono text-2xs text-faint">M{i + 1}</span>
                            {meta.model_names[`m${i + 1}`]}
                          </div>
                          <div className="text-2xs text-mute">{m.plain}</div>
                        </td>
                        <td className="py-2 text-right font-cond tnum">{row?.rmse24.toFixed(2) ?? '—'}</td>
                        <td className="py-2 pl-3 text-right">
                          <WeightBar v={a} />
                        </td>
                        <td className="py-2 pl-3 text-right">
                          <WeightBar v={b} />
                        </td>
                      </tr>
                    )
                  })}
                  <tr className="border-t border-line-strong">
                    <td className="py-2 font-medium">Stacked blend of all eight</td>
                    <td className="py-2 text-right font-cond font-semibold tnum">{w24.oof_rmse_after_shrinkage.toFixed(2)}</td>
                    <td className="py-2 pl-3 text-right text-2xs text-mute" colSpan={2}>
                      shrinkage α {w24.shrinkage_alpha} / {w72.shrinkage_alpha}
                    </td>
                  </tr>
                </tbody>
              </table>
            </section>

            <section>
              <h3 className="text-[15px] font-medium">Reading the map</h3>
              <ul className="mt-2 space-y-1.5 text-[13px] text-mute">
                <li>Column height and colour show PM2.5 at each station for the selected hour, on the CPCB index scale.</li>
                <li>The coloured haze over the city is an inverse-distance blend of 18 stations: the 4 forecast stations and 14 other monitors scaled from them.</li>
                <li>Wind trails follow Open-Meteo 10 m wind at the four stations and observed winds at Mumbai airport (VABB).</li>
                <li>Ward boundaries are approximate cells around BMC ward offices, clipped to the city outline.</li>
              </ul>
            </section>

            <section className="text-2xs leading-relaxed text-faint">
              Basemap © OpenStreetMap contributors via OpenFreeMap. Terrain from Mapzen/AWS Terrain Tiles. Weather from Open-Meteo, chemistry
              from CAMS, airport observations from VABB METAR, population from WorldPop, schools from UDISE+, all via the ClimateVerse India
              Dataverse.
            </section>
          </div>
        ) : (
          <Catalog />
        )}
      </div>
    </>
  )
}

function Metric({ value, label }: { value: number; label: string }) {
  return (
    <div className="panel-solid px-4 py-3">
      <div className="flex items-baseline gap-1.5">
        <span className="font-cond text-[40px] font-semibold leading-none tnum">{value.toFixed(2)}</span>
        <span className="text-xs text-mute">µg/m³</span>
      </div>
      <div className="mt-1.5 text-2xs text-mute">{label}</div>
      <div className="font-mono text-[10px] text-faint tnum">{value.toFixed(4)}</div>
    </div>
  )
}

function WeightBar({ v }: { v: number }) {
  return (
    <div className="ml-auto flex w-[76px] items-center justify-end gap-1.5">
      <div className="h-1 flex-1 overflow-hidden rounded-full bg-paper/8">
        <div className="h-full rounded-full bg-paper/70" style={{ width: `${Math.min(100, v * 100)}%` }} />
      </div>
      <span className="w-7 text-right font-cond text-xs tnum text-mute">{Math.round(v * 100)}%</span>
    </div>
  )
}

function Catalog() {
  const data = useData()
  const [q, setQ] = useState('')
  const [onlyUsed, setOnlyUsed] = useState(true)
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase()
    return data.catalog.datasets
      .filter((d) => (!onlyUsed || d.integrated) && (!t || `${d.name} ${d.description} ${d.doi} ${d.category}`.toLowerCase().includes(t)))
      .sort((a, b) => Number(b.integrated) - Number(a.integrated))
  }, [data, q, onlyUsed])

  return (
    <div>
      <p className="text-[13px] text-mute">
        {data.catalog.meta.integrated} of {data.catalog.meta.count} ClimateVerse India datasets feed the forecasts and the ward tools.
      </p>
      <div className="mt-3 flex items-center gap-2">
        <label className="flex flex-1 items-center gap-2 rounded-[10px] border border-line bg-ink px-3 py-1.5">
          <Search size={14} className="text-faint" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search datasets"
            className="w-full bg-transparent text-[13px] text-paper placeholder:text-faint focus:outline-none"
          />
        </label>
        <Segmented
          size="sm"
          value={onlyUsed ? 'used' : 'all'}
          onChange={(v) => setOnlyUsed(v === 'used')}
          options={[
            { value: 'used', label: 'Used' },
            { value: 'all', label: 'All' },
          ]}
        />
      </div>
      <ul className="mt-4 divide-y divide-line">
        {rows.map((d) => (
          <li key={d.doi} className="py-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className={cx('text-[13px] leading-snug', d.integrated ? 'text-paper' : 'text-mute')}>{d.name}</div>
                <div className="mt-0.5 text-2xs text-faint">{d.curated_category ?? d.category}</div>
              </div>
              <a
                href={d.url}
                target="_blank"
                rel="noreferrer"
                className="flex shrink-0 items-center gap-1 font-mono text-[10px] text-mute hover:text-paper"
              >
                {d.doi.replace('doi:', '')} <ExternalLink size={11} />
              </a>
            </div>
            {d.role && <p className="mt-1 text-xs text-mute">{d.role}</p>}
          </li>
        ))}
        {rows.length === 0 && <li className="py-6 text-center text-[13px] text-mute">No datasets match “{q}”. Try a broader word such as “wind” or “school”.</li>}
      </ul>
    </div>
  )
}
