import type { TwinData } from '../data/types'
import { sampleAt, stationPM } from './series'

const R = 6371

export function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number) {
  const p1 = (lat1 * Math.PI) / 180
  const p2 = (lat2 * Math.PI) / 180
  const dp = p2 - p1
  const dl = ((lon2 - lon1) * Math.PI) / 180
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(a))
}

export interface WindSampler {
  /** returns (u, v) in m/s at lon/lat for the sampler's hour */
  at: (lon: number, lat: number) => [number, number]
  mean: [number, number]
}

/** IDW wind field from 4 Open-Meteo station points + VABB METAR, interpolated between hours. */
export function windSampler(data: TwinData, f: number): WindSampler {
  const pts = Object.values(data.wind.points)
    .map((p) => ({ lat: p.lat, lon: p.lon, u: sampleAt(p.u, f), v: sampleAt(p.v, f), w: p.observed ? 0.7 : 1 }))
    .filter((p): p is { lat: number; lon: number; u: number; v: number; w: number } => p.u != null && p.v != null)
  const mu = pts.reduce((s, p) => s + p.u, 0) / Math.max(1, pts.length)
  const mv = pts.reduce((s, p) => s + p.v, 0) / Math.max(1, pts.length)
  return {
    mean: [mu, mv],
    at(lon, lat) {
      let su = 0
      let sv = 0
      let sw = 0
      for (const p of pts) {
        const dx = (lon - p.lon) * 105
        const dy = (lat - p.lat) * 111
        const w = p.w / (dx * dx + dy * dy + 4)
        su += w * p.u
        sv += w * p.v
        sw += w
      }
      return sw ? [su / sw, sv / sw] : [mu, mv]
    },
  }
}

export interface CityMet {
  pblh: number
  wind: number
  dirFrom: number
  vc: number
  onshore: number
  deonar: number
  sw: number
  regime: string
}

export function cityMet(data: TwinData, f: number): CityMet {
  const c = data.wind.city
  const i = Math.round(f)
  return {
    pblh: sampleAt(c.pblh, f) ?? 400,
    wind: sampleAt(c.wind_ms, f) ?? 2,
    dirFrom: c.dir_from[i] ?? 0,
    vc: sampleAt(c.vc, f) ?? 0,
    onshore: sampleAt(c.onshore, f) ?? 0,
    deonar: sampleAt(c.deonar_transport, f) ?? 0,
    sw: sampleAt(c.sw, f) ?? 0,
    regime: c.regime[i] ?? 'unknown',
  }
}

export const REGIME_LABEL: Record<string, string> = {
  sea_breeze: 'Arabian Sea breeze · clearing',
  land_breeze: 'Land breeze · Deonar / Thane Creek drainage',
  transitional: 'Transitional synoptic flow',
  stagnant: 'Stagnant · inversion trapping',
  unknown: 'No met data',
}

export interface FluxEdge {
  src: number
  dst: number
  flux: number
  km: number
}

/**
 * AirPhyNet-style advection–diffusion flux on the station graph:
 * flux_ij = C_i · max(0, cos(θ_to − bearing_ij)) · |U| / d_ij − κ (C_i − C_j) / d_ij²  (clipped ≥ 0)
 */
export function fluxEdges(data: TwinData, f: number, estimate: 'mean' | 'median' = 'mean'): FluxEdge[] {
  const met = cityMet(data, f)
  const toDeg = (met.dirFrom + 180) % 360
  const kappa = data.wind.airphynet.kappa
  const pm = new Map<number, number>()
  for (const s of data.stations.stations) pm.set(s.id, stationPM(data, s.id, f, estimate) ?? 0)
  const out: FluxEdge[] = []
  for (const e of data.wind.airphynet.edges) {
    const ci = pm.get(e.src) ?? 0
    const cj = pm.get(e.dst) ?? 0
    const align = Math.max(0, Math.cos(((toDeg - e.bearing) * Math.PI) / 180))
    const adv = (ci * align * met.wind) / Math.max(0.8, e.km)
    const diff = (kappa * (ci - cj)) / Math.max(0.8, e.km) ** 2
    const flux = adv + Math.max(0, diff)
    if (flux > 0.5) out.push({ src: e.src, dst: e.dst, flux, km: e.km })
  }
  return out.sort((a, b) => b.flux - a.flux)
}

/** Simple solar elevation for Mumbai (deg) from IST hour and day-of-year. */
export function solarElevation(iso: string): number {
  const d = new Date(iso)
  const start = new Date(d.getFullYear(), 0, 0)
  const doy = Math.floor((d.getTime() - start.getTime()) / 86400000)
  const hour = Number(iso.slice(11, 13)) + Number(iso.slice(14, 16)) / 60
  const decl = 23.44 * Math.sin(((2 * Math.PI) / 365) * (doy - 81))
  const solarTime = hour + (72.87 - 82.5) / 15
  const ha = 15 * (solarTime - 12)
  const lat = 19.07
  const rad = Math.PI / 180
  const el = Math.asin(Math.sin(lat * rad) * Math.sin(decl * rad) + Math.cos(lat * rad) * Math.cos(decl * rad) * Math.cos(ha * rad))
  return el / rad
}
