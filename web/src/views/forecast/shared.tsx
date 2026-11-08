import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { Forecasts } from '../../data/types'

export const C = {
  paper: '#ECE7DC',
  mute: '#9AA4AF',
  faint: '#5F6B78',
  ink: '#0A1119',
  ink2: '#0F1821',
  ink3: '#16212C',
  grid: 'rgba(236,231,220,0.07)',
  gridStrong: 'rgba(236,231,220,0.16)',
}

export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** "11 Nov" from an IST ISO string or YYYY-MM-DD. */
export function dayLabel(iso: string): string {
  const [, m, d] = iso.slice(0, 10).split('-').map(Number)
  return `${d} ${MONTHS[m - 1]}`
}

/** "3 Dec 2025, 13:00" from an IST ISO string. */
export function longStamp(iso: string): string {
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number)
  return `${d} ${MONTHS[m - 1]} ${y}, ${iso.slice(11, 16)}`
}

/** Width of an element, tracked with ResizeObserver. */
export function useWidth<T extends HTMLElement>(initial = 800) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(initial)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(el.getBoundingClientRect().width)
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width
      if (w) setWidth(w)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(mq.matches)
    const on = () => setReduced(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return reduced
}

export function SectionHead({
  eyebrow,
  title,
  subtitle,
  right,
}: {
  eyebrow?: string
  title: string
  subtitle?: ReactNode
  right?: ReactNode
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
      <div className="min-w-0 max-w-[720px]">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h2 className="text-[17px] font-semibold leading-snug tracking-tight text-paper">{title}</h2>
        {subtitle && <p className="mt-1 text-[13px] leading-relaxed text-mute">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

/** Full ensemble-weights record as exported (the shared type only declares a subset). */
export interface GapWeights {
  weights: Record<string, number>
  shrinkage_alpha: number
  intercept_shift?: number
  station_alphas?: Record<string, number>
  station_shifts?: Record<string, number>
  oof_rmse_before_shrinkage?: number
  oof_rmse_after_shrinkage: number
}

export function gapWeights(meta: Forecasts['meta'], gap: 'gap24h' | 'gap72h'): GapWeights | null {
  return (meta.ensemble_weights[gap] as GapWeights | undefined) ?? null
}

/** Model keys used in the per-hour series, in display order. */
export const OVERLAY_KEYS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'timesfm_mean', 'chronos_mean'] as const
export type OverlayKey = (typeof OVERLAY_KEYS)[number]

export const SHORT_MODEL: Record<string, string> = {
  m1: 'CAMS MOS prior',
  m2: 'Physics GAM',
  m3: 'LightGBM',
  m4: 'CatBoost',
  m5: 'XGBoost',
  m6: 'Wind ST-GAT',
  m7: 'TimesFM 3.0',
  m8: 'Chronos-2',
  timesfm_mean: 'TimesFM decile mean',
  chronos_mean: 'Chronos-2 quantile mean',
  ensemble: 'Stacked ensemble',
}

/** Leaderboard / weight key ("m4_catboost_residual") → series key ("m4"). */
export function seriesKey(lbKey: string): string | null {
  const m = /^m(\d)_/.exec(lbKey)
  return m ? `m${m[1]}` : null
}

export const BASELINE_SHORT: Record<string, string> = {
  b0_station_mean_90d: '90-day station mean',
  b1_starter_rf_90d: 'Starter random forest',
  b2_diurnal_shrinkage_7d: '7-day diurnal shrinkage',
  m7_stacked_ensemble: '8-model stacked ensemble',
}

export function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-[160px] items-center justify-center rounded-xl border border-dashed border-line px-6 py-8 text-center text-[13px] leading-relaxed text-mute">
      <div className="max-w-sm">{children}</div>
    </div>
  )
}
