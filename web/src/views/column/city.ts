import * as THREE from 'three'
import { bearingVec, COAST_NORMAL_DEG, M_PER_UNIT, TILE_R } from './derive'

/** Radius of the procedurally built-up area (2.6 km). */
export const CITY_EXTENT = 26
/** Radius of the ground disc; the outer ring fades into the horizon. */
export const GROUND_R = 34
const GREEN_TEX = 160

export interface CityInput {
  seed: number
  density: number
  roughness: number
  green: number
  coastKm: number | null
}

export interface InstanceSet {
  count: number
  matrices: Float32Array
  tones: Float32Array
}

export interface CityLayout {
  buildings: InstanceSet
  trees: InstanceSet
  greenTex: THREE.DataTexture
  coastDir: [number, number]
  coastDist: number | null
  maxBuildingM: number
}

function mulberry32(seed: number) {
  let a = seed | 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hash2(ix: number, iz: number, seed: number) {
  let h = (Math.imul(ix, 374761393) + Math.imul(iz, 668265263) + Math.imul(seed, 1442695041)) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function vnoise(x: number, z: number, seed: number) {
  const ix = Math.floor(x)
  const iz = Math.floor(z)
  const fx = x - ix
  const fz = z - iz
  const sx = fx * fx * (3 - 2 * fx)
  const sz = fz * fz * (3 - 2 * fz)
  const a = hash2(ix, iz, seed)
  const b = hash2(ix + 1, iz, seed)
  const c = hash2(ix, iz + 1, seed)
  const d = hash2(ix + 1, iz + 1, seed)
  return a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz
}

function fbm(x: number, z: number, seed: number) {
  const f = 0.16
  return 0.62 * vnoise(x * f, z * f, seed) + 0.28 * vnoise(x * f * 2.3 + 5.3, z * f * 2.3 - 1.7, seed + 7) + 0.1 * vnoise(x * f * 5.1, z * f * 5.1, seed + 13)
}

/**
 * Stylised neighbourhood: building count and height mix follow Open Buildings density and AlphaEarth
 * roughness, park area follows the AlphaEarth green index, and the sea sits at the true coastal distance.
 */
export function buildCity(inp: CityInput): CityLayout {
  const rng = mulberry32(inp.seed)
  const seed = inp.seed % 100000
  const coastDir = bearingVec(COAST_NORMAL_DEG)
  const coastDist = inp.coastKm != null ? (inp.coastKm * 1000) / M_PER_UNIT : null
  const isWater = (x: number, z: number, margin = 0) =>
    coastDist != null && x * coastDir[0] + z * coastDir[1] > coastDist - margin

  // Green threshold: the noise quantile that makes the park share inside the tile equal the green index.
  const samples: number[] = []
  for (let x = -TILE_R; x <= TILE_R; x += 0.5)
    for (let z = -TILE_R; z <= TILE_R; z += 0.5) if (x * x + z * z <= TILE_R * TILE_R) samples.push(fbm(x, z, seed))
  samples.sort((p, q) => p - q)
  const g = Math.max(0, Math.min(0.95, inp.green))
  const thr = g <= 0 ? -1 : samples[Math.min(samples.length - 1, Math.floor(g * samples.length))]
  const isGreen = (x: number, z: number) => fbm(x, z, seed) < thr

  const tex = new Uint8Array(GREEN_TEX * GREEN_TEX)
  for (let j = 0; j < GREEN_TEX; j++)
    for (let i = 0; i < GREEN_TEX; i++) {
      const x = -GROUND_R + ((i + 0.5) * 2 * GROUND_R) / GREEN_TEX
      const z = -GROUND_R + ((j + 0.5) * 2 * GROUND_R) / GREEN_TEX
      const n = fbm(x, z, seed)
      const t = Math.max(0, Math.min(1, (thr + 0.025 - n) / 0.05))
      tex[j * GREEN_TEX + i] = Math.round(t * 255)
    }
  const greenTex = new THREE.DataTexture(tex, GREEN_TEX, GREEN_TEX, THREE.RedFormat, THREE.UnsignedByteType)
  greenTex.magFilter = THREE.LinearFilter
  greenTex.minFilter = THREE.LinearFilter
  greenTex.generateMipmaps = false
  greenTex.needsUpdate = true

  const cell = 0.56
  const angle = (rng() - 0.5) * 0.7
  const ca = Math.cos(angle)
  const sa = Math.sin(angle)
  const n = Math.ceil(CITY_EXTENT / cell)
  const bMat: number[] = []
  const bTone: number[] = []
  const tMat: number[] = []
  const tTone: number[] = []
  const m4 = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const yAxis = new THREE.Vector3(0, 1, 0)
  const pos = new THREE.Vector3()
  const scl = new THREE.Vector3()
  const towerP = 0.03 + 0.22 * inp.roughness * inp.density
  const midP = 0.18 + 0.3 * inp.roughness
  const occupy = 0.3 + 0.65 * inp.density
  let maxH = 0
  const MAX_TREES = 5200

  for (let i = -n; i <= n; i++) {
    for (let j = -n; j <= n; j++) {
      // Arterial roads every few blocks keep the grid readable from above.
      if (i % 7 === 0 || j % 6 === 0) continue
      const gx = i * cell
      const gz = j * cell
      const x = gx * ca - gz * sa + (rng() - 0.5) * 0.08
      const z = gx * sa + gz * ca + (rng() - 0.5) * 0.08
      const r = Math.hypot(x, z)
      if (r > CITY_EXTENT || r < 0.75) continue
      if (isWater(x, z, 0.5)) continue
      const outer = r > TILE_R
      if (isGreen(x, z)) {
        if (r > 22 || tMat.length / 16 >= MAX_TREES) continue
        const k = 1 + Math.floor(rng() * (outer ? 2 : 3))
        for (let t = 0; t < k; t++) {
          const s = 0.08 + rng() * 0.09
          pos.set(x + (rng() - 0.5) * cell * 0.8, s * 0.7, z + (rng() - 0.5) * cell * 0.8)
          q.setFromAxisAngle(yAxis, rng() * Math.PI)
          scl.set(s, s * (0.75 + rng() * 0.3), s)
          m4.compose(pos, q, scl)
          tMat.push(...m4.elements)
          tTone.push(0.8 + rng() * 0.4)
        }
        continue
      }
      const fall = outer ? 0.75 * (1 - ((r - TILE_R) / (CITY_EXTENT - TILE_R)) * 0.45) : 1
      if (rng() > occupy * fall) continue
      const roll = rng()
      let hM: number
      let w: number
      let d: number
      const tp = outer ? towerP * 0.6 : towerP
      if (roll < tp) {
        hM = 60 + Math.pow(rng(), 1.6) * (70 + 160 * inp.roughness)
        w = 0.2 + rng() * 0.14
        d = w * (0.8 + rng() * 0.4)
      } else if (roll < tp + midP) {
        hM = 18 + rng() * 26
        w = 0.24 + rng() * 0.2
        d = 0.24 + rng() * 0.2
      } else {
        hM = 6 + rng() * 12
        w = 0.28 + rng() * 0.22
        d = 0.28 + rng() * 0.2
      }
      if (r > CITY_EXTENT - 4) hM *= (CITY_EXTENT - r) / 4
      const h = Math.max(0.04, hM / M_PER_UNIT)
      maxH = Math.max(maxH, hM)
      pos.set(x, 0, z)
      q.setFromAxisAngle(yAxis, angle + (rng() - 0.5) * 0.12)
      scl.set(w, h, d)
      m4.compose(pos, q, scl)
      bMat.push(...m4.elements)
      bTone.push(0.82 + rng() * 0.36)
    }
  }

  return {
    buildings: { count: bTone.length, matrices: new Float32Array(bMat), tones: new Float32Array(bTone) },
    trees: { count: tTone.length, matrices: new Float32Array(tMat), tones: new Float32Array(tTone) },
    greenTex,
    coastDir,
    coastDist,
    maxBuildingM: maxH,
  }
}
