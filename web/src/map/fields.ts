import type { TwinData } from '../data/types'
import { pmRGB } from '../lib/aqi'
import { windSampler } from '../lib/physics'
import { sampleAt, stationPM } from '../lib/series'

/** Geographic extent of the interpolated PM2.5 surface (covers the BMC outline). */
export const SURFACE_BOUNDS = { west: 72.76, east: 73.01, south: 18.88, north: 19.31 }
const GRID_W = 150
const GRID_H = 260

/** Image corners in MapLibre `image` source order: TL, TR, BR, BL. */
export const SURFACE_COORDS: [[number, number], [number, number], [number, number], [number, number]] = [
  [SURFACE_BOUNDS.west, SURFACE_BOUNDS.north],
  [SURFACE_BOUNDS.east, SURFACE_BOUNDS.north],
  [SURFACE_BOUNDS.east, SURFACE_BOUNDS.south],
  [SURFACE_BOUNDS.west, SURFACE_BOUNDS.south],
]

export interface StationValue {
  id: number
  lon: number
  lat: number
  pm: number
}

/** PM2.5 at hour f for all 18 stations: ensemble forecast for targets, calibrated series for the rest. */
export function stationValues(data: TwinData, f: number): StationValue[] {
  const out: StationValue[] = []
  for (const s of data.stations.stations) {
    const pm = s.role === 'target' ? stationPM(data, s.id, f) : (sampleAt(s.series, f) ?? sampleAt(s.obs, f))
    if (pm != null && Number.isFinite(pm)) out.push({ id: s.id, lon: s.lon, lat: s.lat, pm })
  }
  return out
}

let maskCanvas: HTMLCanvasElement | null = null

function landMask(outline: [number, number][]): HTMLCanvasElement {
  if (maskCanvas) return maskCanvas
  const c = document.createElement('canvas')
  c.width = GRID_W
  c.height = GRID_H
  const ctx = c.getContext('2d')!
  const { west, east, south, north } = SURFACE_BOUNDS
  ctx.filter = 'blur(5px)'
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  outline.forEach(([lon, lat], i) => {
    const x = ((lon - west) / (east - west)) * GRID_W
    const y = ((north - lat) / (north - south)) * GRID_H
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.closePath()
  ctx.fill()
  maskCanvas = c
  return c
}

/**
 * Inverse-distance-weighted PM2.5 surface (power 2, 0.6 km smoothing radius) rendered to a
 * small canvas, coloured with the CPCB ramp and masked to the BMC outline.
 */
export function renderSurface(data: TwinData, f: number, target: HTMLCanvasElement): void {
  const values = stationValues(data, f)
  target.width = GRID_W
  target.height = GRID_H
  const ctx = target.getContext('2d')!
  const img = ctx.createImageData(GRID_W, GRID_H)
  const { west, east, south, north } = SURFACE_BOUNDS
  const kx = 105.2
  const ky = 110.6
  for (let j = 0; j < GRID_H; j++) {
    const lat = north - ((j + 0.5) / GRID_H) * (north - south)
    for (let i = 0; i < GRID_W; i++) {
      const lon = west + ((i + 0.5) / GRID_W) * (east - west)
      let sw = 0
      let sv = 0
      for (const p of values) {
        const dx = (lon - p.lon) * kx
        const dy = (lat - p.lat) * ky
        const w = 1 / (dx * dx + dy * dy + 0.36)
        sw += w
        sv += w * p.pm
      }
      const pm = sv / sw
      const [r, g, b] = pmRGB(pm)
      const o = (j * GRID_W + i) * 4
      img.data[o] = r
      img.data[o + 1] = g
      img.data[o + 2] = b
      img.data[o + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  ctx.globalCompositeOperation = 'destination-in'
  ctx.drawImage(landMask(data.wards.outline), 0, 0)
  ctx.globalCompositeOperation = 'source-over'
}

export interface Streamline {
  path: [number, number][]
  timestamps: number[]
  speed: number
}

/** Length of one animation loop in streamline time units. */
export const WIND_LOOP = 60

function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Streamlines traced through the IDW wind field (4 Open-Meteo points + VABB METAR) for hour f.
 * Each line advects for `STEPS` steps; step length scales with local wind speed so faster
 * air draws longer trails. Lines are duplicated one loop earlier so the animation wraps seamlessly.
 */
export function traceStreamlines(data: TwinData, f: number, count = 900): Streamline[] {
  const field = windSampler(data, f)
  const rand = mulberry32(1337)
  const STEPS = 28
  // Seconds of real advection per step, exaggerated so 2 m/s air moves visibly.
  const DT = 70
  const out: Streamline[] = []
  const ext = { west: 72.68, east: 73.08, south: 18.84, north: 19.34 }
  for (let n = 0; n < count; n++) {
    let lon = ext.west + rand() * (ext.east - ext.west)
    let lat = ext.south + rand() * (ext.north - ext.south)
    const offset = rand() * WIND_LOOP
    const path: [number, number][] = [[lon, lat]]
    const timestamps: number[] = [offset]
    let speedSum = 0
    for (let s = 1; s <= STEPS; s++) {
      const [u, v] = field.at(lon, lat)
      const sp = Math.hypot(u, v)
      speedSum += sp
      lon += (u * DT) / (111320 * Math.cos((lat * Math.PI) / 180))
      lat += (v * DT) / 110540
      path.push([lon, lat])
      timestamps.push(offset + s)
    }
    const speed = speedSum / STEPS
    out.push({ path, timestamps, speed })
    out.push({ path, timestamps: timestamps.map((t) => t - WIND_LOOP), speed })
  }
  return out
}
