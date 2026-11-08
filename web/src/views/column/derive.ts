import type { TwinData } from '../../data/types'
import { pmRGB } from '../../lib/aqi'
import { haversineKm, REGIME_LABEL, solarElevation } from '../../lib/physics'
import { sampleAt, stationPM } from '../../lib/series'
import { istParts, TARGET_META } from '../../lib/stations'

/** Scene scale: 1 unit = 100 m on every axis (no vertical exaggeration). */
export const M_PER_UNIT = 100
/** Neighbourhood tile radius (1 km). */
export const TILE_R = 10
/** Radius of the air column drawn above the monitor (550 m). */
export const COL_R = 5.5
/** Lowest lid drawn, so a collapsed mixing layer stays visible. */
export const MIN_VIS_M = 30
/** Wind drift speed-up for animation: 1 m/s → 0.45 units/s, i.e. 45× real time. */
export const DRIFT_PER_MS = 0.45
export const TIME_LAPSE = Math.round(DRIFT_PER_MS * M_PER_UNIT)
export const AXIS_TICKS_M = [0, 250, 500, 1000, 1500, 2000, 2500, 3000]
export const COAST_NORMAL_DEG = 285

const RAD = Math.PI / 180
const COMPASS16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

export function windFromDeg(u: number, v: number): number {
  return (Math.atan2(-u, -v) / RAD + 360) % 360
}

export function compass16(deg: number): string {
  return COMPASS16[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16]
}

/** Scene-plane unit vector (x = east, z = south) pointing toward a compass bearing. */
export function bearingVec(deg: number): [number, number] {
  return [Math.sin(deg * RAD), -Math.cos(deg * RAD)]
}

function angDiff(a: number, b: number) {
  const d = Math.abs((((a - b) % 360) + 360) % 360)
  return d > 180 ? 360 - d : d
}

/** Solar azimuth (deg clockwise from north) for Mumbai, same model as `solarElevation`. */
export function solarAzimuth(iso: string): number {
  const d = new Date(iso)
  const start = new Date(d.getFullYear(), 0, 0)
  const doy = Math.floor((d.getTime() - start.getTime()) / 86400000)
  const hour = Number(iso.slice(11, 13)) + Number(iso.slice(14, 16)) / 60
  const decl = 23.44 * Math.sin(((2 * Math.PI) / 365) * (doy - 81)) * RAD
  const ha = 15 * (hour + (72.87 - 82.5) / 15 - 12)
  const lat = 19.07 * RAD
  const el = solarElevation(iso) * RAD
  const cosAz = (Math.sin(decl) - Math.sin(el) * Math.sin(lat)) / Math.max(1e-6, Math.cos(el) * Math.cos(lat))
  const az = Math.acos(Math.max(-1, Math.min(1, cosAz))) / RAD
  return ha > 0 ? 360 - az : az
}

export function fmtInt(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? '—' : Math.round(v).toLocaleString('en-IN')
}

export function fmtRatio(r: number): string {
  return r >= 10 ? `${Math.round(r)}×` : `${r.toFixed(1)}×`
}

/** Normalised vertical PM2.5 profile C(z) / C0, with a sharp change at the mixing lid. */
export function verticalProfileFraction(zM: number, lidM: number, stratification: number): number {
  const z = Math.max(0, zM)
  const lid = Math.max(1, lidM)
  if (z <= lid) return Math.max(0, 1 - 0.35 * Math.pow(z / lid, Math.max(0.1, stratification)))
  // The clean free troposphere decays over four scene units (400 m) above the inversion.
  return 0.12 * Math.exp(-(z - lid) / (4 * M_PER_UNIT))
}

export interface HourMet {
  pblh: number | null
  u: number | null
  v: number | null
  speed: number | null
  dirFrom: number | null
  vent: number | null
  sw: number | null
  usedCity: boolean
}

export function stationMet(data: TwinData, id: number, i: number): HourMet {
  const p = data.wind.points[String(id)]
  const c = data.wind.city
  const own = p?.pblh?.[i]
  const pblh = own ?? c.pblh[i] ?? null
  const u = p?.u[i] ?? c.u[i] ?? null
  const v = p?.v[i] ?? c.v[i] ?? null
  const speed = u != null && v != null ? Math.hypot(u, v) : (c.wind_ms[i] ?? null)
  const dirFrom = u != null && v != null && speed! > 0.05 ? windFromDeg(u, v) : (c.dir_from[i] ?? null)
  const vent = pblh != null && speed != null ? pblh * speed : (c.vc[i] ?? null)
  const sw = p?.sw?.[i] ?? c.sw[i] ?? null
  return { pblh, u, v, speed, dirFrom, vent, sw, usedCity: own == null }
}

export interface Peak {
  i: number
  value: number
  time: string
}

export interface NearbySource {
  name: string
  km: number
  bearing: number
}

export interface ColumnState {
  i: number
  iso: string
  parts: ReturnType<typeof istParts>
  stationName: string
  area: string
  pm: number | null
  pmSource: 'Ensemble forecast' | 'Measured' | null
  measured: number | null
  met: HourMet
  dirText: string | null
  exposure: 'off the sea' | 'off the land' | 'along the coast' | null
  regime: string
  regimeKey: string
  dayPeak: Peak | null
  dayVentPeak: Peak | null
  sentence: string
  sunEl: number
  sunAz: number
  morphology: { density: number; roughness: number; green: number; coastKm: number | null }
  sources: NearbySource[]
}

function pblhAt(data: TwinData, id: number, i: number) {
  return data.wind.points[String(id)]?.pblh?.[i] ?? data.wind.city.pblh[i] ?? null
}

function dayRange(data: TwinData, i: number): [number, number] {
  const hours = data.forecasts.hours
  const day = hours[i].slice(0, 10)
  let a = i
  let b = i
  while (a > 0 && hours[a - 1].slice(0, 10) === day) a--
  while (b < hours.length - 1 && hours[b + 1].slice(0, 10) === day) b++
  return [a, b]
}

function peakOver(data: TwinData, a: number, b: number, f: (k: number) => number | null): Peak | null {
  let best: Peak | null = null
  for (let k = a; k <= b; k++) {
    const val = f(k)
    if (val != null && Number.isFinite(val) && (!best || val > best.value)) {
      best = { i: k, value: val, time: data.forecasts.hours[k].slice(11, 16) }
    }
  }
  return best
}

function exposureOf(dirFrom: number | null): ColumnState['exposure'] {
  if (dirFrom == null) return null
  const toSea = angDiff(dirFrom, COAST_NORMAL_DEG)
  if (toSea <= 60) return 'off the sea'
  if (toSea >= 120) return 'off the land'
  return 'along the coast'
}

function buildSentence(time: string, met: HourMet, peak: Peak | null, i: number): string {
  const h = met.pblh
  if (h == null) return `Mixing-height data is missing at ${time}, so the column is drawn from the city average.`
  let s: string
  if (h < 50) {
    s = `At ${time} the mixing layer has collapsed to under 50 m, so emissions are held right at street level`
    if (peak && peak.value > 200) {
      s += peak.i > i ? `. By ${peak.time} it will have grown to ${fmtInt(peak.value)} m.` : `. At ${peak.time} it was ${fmtInt(peak.value)} m deep.`
    } else s += '.'
  } else if (!peak || peak.value <= h * 1.15) {
    s = `At ${time} the mixing layer is ${fmtInt(h)} m deep, close to the day's highest, so emissions are spread through a tall column of air.`
  } else {
    const ratio = peak.value / h
    s =
      ratio >= 1.5
        ? `At ${time} the mixing layer is ${fmtInt(h)} m deep, ${fmtRatio(ratio)} shallower than at ${peak.time}, so the same emissions are packed into a thinner column.`
        : `At ${time} the mixing layer is ${fmtInt(h)} m deep, a little below the day's ${fmtInt(peak.value)} m at ${peak.time}.`
  }
  const w = met.speed
  if (w != null) {
    if (w < 1) s += ' The air is almost calm, so little of it is carried away.'
    else if (w < 2.5) s += ` A light ${w.toFixed(1)} m/s wind moves it only slowly.`
    else s += ` A ${w.toFixed(1)} m/s wind helps carry it away.`
  }
  return s
}

export function deriveColumn(data: TwinData, id: number, i: number): ColumnState {
  const iso = data.forecasts.hours[i]
  const parts = istParts(iso)
  const st = data.stations.stations.find((s) => s.id === id)
  const fc = data.forecasts.stations[String(id)]
  const ens = fc ? sampleAt(fc.models.ensemble, i) : null
  const pm = stationPM(data, id, i)
  const measured = fc?.obs[i] ?? null
  const met = stationMet(data, id, i)
  const [a, b] = dayRange(data, i)
  const dayPeak = peakOver(data, a, b, (k) => pblhAt(data, id, k))
  const dayVentPeak = peakOver(data, a, b, (k) => stationMet(data, id, k).vent)
  const regimeKey = data.wind.city.regime[i] ?? 'unknown'
  const m = st?.morphology ?? {}
  const num = (k: string, d: number) => {
    const v = m[k]
    return v != null && Number.isFinite(v) ? v : d
  }
  const coast = m.coastal_dist_km
  const sources: NearbySource[] = []
  if (st) {
    for (const p of data.wind.plume_sources) {
      const km = haversineKm(st.lat, st.lon, p.lat, p.lon)
      if (km > 2.6) continue
      const y = Math.sin((p.lon - st.lon) * RAD) * Math.cos(p.lat * RAD)
      const x = Math.cos(st.lat * RAD) * Math.sin(p.lat * RAD) - Math.sin(st.lat * RAD) * Math.cos(p.lat * RAD) * Math.cos((p.lon - st.lon) * RAD)
      const bearing = (Math.atan2(y, x) / RAD + 360) % 360
      sources.push({ name: shortSourceName(p.name), km, bearing })
    }
  }
  return {
    i,
    iso,
    parts,
    stationName: TARGET_META[id]?.name ?? st?.short ?? String(id),
    area: TARGET_META[id]?.area ?? '',
    pm,
    pmSource: ens != null ? 'Ensemble forecast' : pm != null ? 'Measured' : null,
    measured,
    met,
    dirText: met.dirFrom != null ? compass16(met.dirFrom) : null,
    exposure: exposureOf(met.dirFrom),
    regime: REGIME_LABEL[regimeKey] ?? REGIME_LABEL.unknown,
    regimeKey,
    dayPeak,
    dayVentPeak,
    sentence: buildSentence(parts.time, met, dayPeak, i),
    sunEl: solarElevation(iso),
    sunAz: solarAzimuth(iso),
    morphology: {
      density: num('open_buildings_density', 0.6),
      roughness: num('alphaearth_roughness', 0.6),
      green: num('alphaearth_green_idx', 0.2),
      coastKm: coast != null && Number.isFinite(coast) && coast < 3 ? coast : null,
    },
    sources,
  }
}

function shortSourceName(name: string): string {
  const base = name.split(/\s[(&–]/)[0].trim()
  return base
    .replace(/ Dumping Ground$/, ' landfill')
    .replace(/ \(MbPT\)$/, '')
    .replace(/ Thermal Power Station$/, ' power station')
}

/** Everything the 3D scene animates toward; read inside useFrame. */
export interface SceneTargets {
  lidM: number
  peakM: number | null
  pm: number
  rgb: [number, number, number]
  u: number
  v: number
  conv: number
  sunEl: number
  sunAz: number
}

export function sceneTargets(st: ColumnState): SceneTargets {
  const [r, g, b] = pmRGB(st.pm)
  return {
    lidM: Math.max(MIN_VIS_M, st.met.pblh ?? 400),
    peakM: st.dayPeak?.value ?? null,
    pm: st.pm ?? 0,
    rgb: [r / 255, g / 255, b / 255],
    u: st.met.u ?? 0,
    v: st.met.v ?? 0,
    conv: Math.max(0, Math.min(1, (st.met.sw ?? 0) / 700)),
    sunEl: st.sunEl,
    sunAz: st.sunAz,
  }
}
