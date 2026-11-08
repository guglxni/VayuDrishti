import type { Num, TwinData } from '../data/types'

/** Linear interpolation into a nullable series at fractional index; nearest non-null fallback. */
export function sampleAt(arr: Num[] | undefined, f: number): number | null {
  if (!arr || arr.length === 0) return null
  const i0 = Math.max(0, Math.min(arr.length - 1, Math.floor(f)))
  const i1 = Math.min(arr.length - 1, i0 + 1)
  const a = arr[i0]
  const b = arr[i1]
  if (a != null && b != null) return a + (b - a) * (f - i0)
  if (a != null) return a
  if (b != null) return b
  for (let d = 1; d < 6; d++) {
    const l = arr[i0 - d]
    const r = arr[i0 + d]
    if (l != null) return l
    if (r != null) return r
  }
  return null
}

/** Best available PM2.5 for a station at hour f: forecast ensemble, else observation. */
export function stationPM(data: TwinData, id: number, f: number, estimate: 'mean' | 'median' = 'mean'): number | null {
  const fc = data.forecasts.stations[String(id)]
  if (fc) {
    const key = estimate === 'median' ? fc.median : fc.models.ensemble
    return sampleAt(key, f) ?? sampleAt(fc.obs_clean, f)
  }
  const st = data.stations.stations.find((s) => s.id === id)
  return sampleAt(st?.series, f) ?? sampleAt(st?.obs, f)
}

export function fmtHour(iso: string | undefined, opts: { withDate?: boolean } = { withDate: true }): string {
  if (!iso) return '—'
  const d = iso.slice(0, 10)
  const t = iso.slice(11, 16)
  if (!opts.withDate) return t
  const [, m, day] = d.split('-')
  const months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
  return `${day} ${months[Number(m) - 1]} · ${t} IST`
}

export function hourOfDay(iso: string | undefined): number {
  return iso ? Number(iso.slice(11, 13)) : 12
}

export function fmt(v: number | null | undefined, digits = 1): string {
  return v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits)
}

export function finite(arr: Num[]): number[] {
  return arr.filter((v): v is number => v != null && Number.isFinite(v))
}
