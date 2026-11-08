import { useEffect, useLayoutEffect, useMemo, useRef, type ReactNode, type RefObject } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl, Line2, LineSegments2 } from 'three-stdlib'
import type { Station } from '../../data/types'
import type { FluxEdge } from '../../lib/physics'
import { buildCity, GROUND_R, type CityInput, type CityLayout } from './city'
import {
  AXIS_TICKS_M,
  bearingVec,
  COL_R,
  DRIFT_PER_MS,
  fmtInt,
  M_PER_UNIT,
  TILE_R,
  verticalProfileFraction,
  type NearbySource,
  type SceneTargets,
} from './derive'
import {
  COLUMN_VERT,
  DISC_VERT,
  FOOT_FRAG,
  HAZE_FRAG,
  LID_FRAG,
  PARTICLE_FRAG,
  PARTICLE_VERT,
  SHELL_FRAG,
  SKY_FRAG,
  SKY_VERT,
  STREAK_FRAG,
  STREAK_VERT,
} from './shaders'

const PAPER = new THREE.Vector3(236 / 255, 231 / 255, 220 / 255)
const WIND = new THREE.Vector3(207 / 255, 227 / 255, 242 / 255)
const N_PARTICLES = 5200
const N_CLEAN = 700
const TOP_U = 32
const PROFILE_GRID_POINTS = 65
const PROFILE_POINTS = PROFILE_GRID_POINTS + 2
const PROFILE_TOP_U = AXIS_TICKS_M[AXIS_TICKS_M.length - 1] / M_PER_UNIT
const PROFILE_STEP_HALF_WIDTH_U = 0.01
// Keep HTML label teardown independent of the R3F canvas container on view changes.
const HTML_PORTAL = { current: document.body }
const UP = new THREE.Vector3(0, 1, 0)

/** Smoothed scene state, advanced once per frame by <Driver>. */
interface Cur {
  ready: boolean
  lid: number
  peak: number
  pm: number
  rgb: THREE.Vector3
  u: number
  v: number
  conv: number
  sunDir: THREE.Vector3
  sunEl: number
  day: number
  zenith: THREE.Vector3
  horizon: THREE.Vector3
  drift: THREE.Vector2
  mixPhase: number
  travel: number
  time: number
}

function makeCur(): Cur {
  return {
    ready: false,
    lid: 1,
    peak: -1,
    pm: 0,
    rgb: new THREE.Vector3(0.6, 0.6, 0.6),
    u: 0,
    v: 0,
    conv: 0,
    sunDir: new THREE.Vector3(0, 1, 0),
    sunEl: 0,
    day: 0,
    zenith: new THREE.Vector3(),
    horizon: new THREE.Vector3(),
    drift: new THREE.Vector2(),
    mixPhase: 0,
    travel: 0,
    time: 0,
  }
}

const v3 = (r: number, g: number, b: number) => new THREE.Vector3(r / 255, g / 255, b / 255)
const NIGHT_Z = v3(6, 10, 16)
const NIGHT_H = v3(17, 26, 36)
const DAY_Z = v3(44, 58, 72)
const DAY_H = v3(104, 116, 127)
const TWILIGHT_H = v3(96, 82, 74)
const smooth = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

function sunVector(elDeg: number, azDeg: number, out: THREE.Vector3) {
  const el = (elDeg * Math.PI) / 180
  const az = (azDeg * Math.PI) / 180
  return out.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).normalize()
}

function Driver({ targets, cur }: { targets: RefObject<SceneTargets>; cur: Cur }) {
  const sunT = useMemo(() => new THREE.Vector3(), [])
  useFrame((_, rawDt) => {
    const t = targets.current
    const dt = Math.min(rawDt, 0.1)
    const lidT = t.lidM / M_PER_UNIT
    sunVector(Math.max(t.sunEl, -25), t.sunAz, sunT)
    if (!cur.ready) {
      cur.ready = true
      cur.lid = lidT
      cur.peak = t.peakM != null ? t.peakM / M_PER_UNIT : -1
      cur.pm = t.pm
      cur.rgb.set(...t.rgb)
      cur.u = t.u
      cur.v = t.v
      cur.conv = t.conv
      cur.sunDir.copy(sunT)
    }
    const k = 1 - Math.exp(-dt * 4.5)
    cur.time += dt
    cur.lid += (lidT - cur.lid) * k
    const peakT = t.peakM != null ? t.peakM / M_PER_UNIT : -1
    cur.peak = cur.peak < 0 || peakT < 0 ? peakT : cur.peak + (peakT - cur.peak) * k
    cur.pm += (t.pm - cur.pm) * k
    cur.rgb.x += (t.rgb[0] - cur.rgb.x) * k
    cur.rgb.y += (t.rgb[1] - cur.rgb.y) * k
    cur.rgb.z += (t.rgb[2] - cur.rgb.z) * k
    cur.u += (t.u - cur.u) * k
    cur.v += (t.v - cur.v) * k
    cur.conv += (t.conv - cur.conv) * k
    cur.sunDir.lerp(sunT, k).normalize()
    cur.sunEl = (Math.asin(cur.sunDir.y) * 180) / Math.PI
    cur.day = smooth(-4, 18, cur.sunEl)
    const tw = Math.exp(-(((cur.sunEl - 1) / 5) ** 2))
    cur.horizon.copy(NIGHT_H).lerp(DAY_H, cur.day).lerp(TWILIGHT_H, tw * 0.45)
    cur.zenith.copy(NIGHT_Z).lerp(DAY_Z, cur.day)
    const tint = smooth(30, 150, cur.pm) * (0.1 + 0.22 * cur.day)
    cur.horizon.lerp(cur.rgb, tint)
    cur.zenith.lerp(cur.rgb, tint * 0.35)
    cur.drift.x += cur.u * DRIFT_PER_MS * dt
    cur.drift.y += -cur.v * DRIFT_PER_MS * dt
    cur.mixPhase += (0.035 + 0.3 * cur.conv) * dt
    cur.travel += Math.hypot(cur.u, cur.v) * DRIFT_PER_MS * dt
  }, -1)
  return null
}

function Sky({ cur }: { cur: Cur }) {
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uZenith: { value: new THREE.Vector3() },
          uHorizon: { value: new THREE.Vector3() },
          uSunDir: { value: new THREE.Vector3(0, 1, 0) },
          uSunCol: { value: PAPER.clone() },
          uSun: { value: 0 },
        },
        vertexShader: SKY_VERT,
        fragmentShader: SKY_FRAG,
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
      }),
    [],
  )
  const scene = useThree((s) => s.scene)
  const fog = useMemo(() => new THREE.Fog(0x000000, 70, 190), [])
  useEffect(() => {
    scene.fog = fog
    return () => {
      scene.fog = null
      mat.dispose()
    }
  }, [scene, fog, mat])
  useFrame(() => {
    const u = mat.uniforms
    u.uZenith.value.copy(cur.zenith)
    u.uHorizon.value.copy(cur.horizon)
    u.uSunDir.value.copy(cur.sunDir)
    u.uSun.value = smooth(-3, 4, cur.sunEl)
    fog.color.setRGB(cur.horizon.x, cur.horizon.y, cur.horizon.z, THREE.SRGBColorSpace)
    // Haze: visibility shortens as PM2.5 rises.
    fog.far = 200 - Math.min(90, cur.pm * 0.6)
    fog.near = 62
  })
  return (
    <mesh material={mat} renderOrder={-10} frustumCulled={false}>
      <sphereGeometry args={[500, 48, 24]} />
    </mesh>
  )
}

function Lights({ cur }: { cur: Cur }) {
  const sun = useRef<THREE.DirectionalLight>(null)
  const hemi = useRef<THREE.HemisphereLight>(null)
  const dir = useMemo(() => new THREE.Vector3(), [])
  useFrame(() => {
    const s = sun.current
    const h = hemi.current
    if (!s || !h) return
    dir.copy(cur.sunDir)
    if (dir.y < 0.12) {
      dir.y = 0.12
      dir.normalize()
    }
    s.position.copy(dir).multiplyScalar(45)
    s.intensity = 3.4 * smooth(-1, 12, cur.sunEl)
    s.color.setRGB(1, 0.93 + 0.07 * cur.day, 0.84 + 0.16 * cur.day, THREE.SRGBColorSpace)
    h.intensity = 1.25 + 1.6 * cur.day
    h.color.setRGB(0.42 + 0.4 * cur.day, 0.5 + 0.38 * cur.day, 0.62 + 0.3 * cur.day, THREE.SRGBColorSpace)
  })
  return (
    <>
      <hemisphereLight ref={hemi} args={['#6a7f96', '#0a0f14', 1.4]} />
      <directionalLight
        ref={sun}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-18}
        shadow-camera-right={18}
        shadow-camera-top={18}
        shadow-camera-bottom={-18}
        shadow-camera-near={1}
        shadow-camera-far={110}
        shadow-bias={-0.0004}
        shadow-normalBias={0.03}
      />
    </>
  )
}

function Ground({ city, cur }: { city: CityLayout; cur: Cur }) {
  const uniforms = useMemo(
    () => ({
      uGreen: { value: city.greenTex },
      uExtent: { value: GROUND_R },
      uCoastOn: { value: city.coastDist != null ? 1 : 0 },
      uCoastDir: { value: new THREE.Vector2(...city.coastDir) },
      uCoastDist: { value: city.coastDist ?? 999 },
      uLand: { value: new THREE.Color('#1a232c') },
      uPark: { value: new THREE.Color('#1c2a23') },
      uWater: { value: new THREE.Color('#0d1824') },
      uSand: { value: new THREE.Color('#2a2e30') },
      uHorizonLin: { value: new THREE.Color() },
      uTime: { value: 0 },
      uTileR: { value: TILE_R },
    }),
    [city],
  )
  const mat = useMemo(() => {
    const m = new THREE.MeshLambertMaterial({ color: '#ffffff' })
    m.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms)
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vGXZ;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvGXZ = (modelMatrix * vec4(transformed, 1.0)).xz;')
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
varying vec2 vGXZ;
uniform sampler2D uGreen;
uniform float uExtent, uCoastOn, uCoastDist, uTime, uTileR;
uniform vec2 uCoastDir;
uniform vec3 uLand, uPark, uWater, uSand, uHorizonLin;`,
        )
        .replace(
          'vec4 diffuseColor = vec4( diffuse, opacity );',
          `vec2 gp = vGXZ;
float gr = length(gp);
float green = texture2D(uGreen, gp / (2.0 * uExtent) + 0.5).r;
vec3 gcol = mix(uLand, uPark, green * 0.9);
float sea = 0.0;
float cs = dot(gp, uCoastDir) - uCoastDist;
if (uCoastOn > 0.5) {
  sea = smoothstep(-0.04, 0.04, cs);
  float beach = 1.0 - smoothstep(0.0, 0.32, abs(cs + 0.18));
  gcol = mix(gcol, uSand, beach * 0.7 * (1.0 - sea));
  gcol = mix(gcol, uWater, sea);
}
vec4 diffuseColor = vec4(gcol, opacity);`,
        )
        .replace(
          '#include <opaque_fragment>',
          `float aa = fwidth(gr) * 1.5;
float ring = 1.0 - smoothstep(0.0, aa, abs(gr - uTileR) - 0.015);
outgoingLight += vec3(0.05) * ring;
if (uCoastOn > 0.5) {
  float rip = sin(dot(gp, vec2(1.7, 0.6)) * 2.2 - uTime * 0.9) * sin(dot(gp, vec2(-0.5, 1.4)) * 1.6 + uTime * 0.6);
  outgoingLight += vec3(0.012, 0.016, 0.02) * sea * (0.6 + 0.4 * rip);
  float shore = 1.0 - smoothstep(0.0, 0.06, abs(cs));
  outgoingLight += vec3(0.03) * shore;
}
outgoingLight = mix(outgoingLight, uHorizonLin, smoothstep(uExtent * 0.62, uExtent * 0.98, gr));
#include <opaque_fragment>`,
        )
    }
    m.customProgramCacheKey = () => 'column-ground'
    return m
  }, [uniforms])
  useEffect(() => () => mat.dispose(), [mat])
  useFrame(() => {
    uniforms.uTime.value = cur.time
    uniforms.uHorizonLin.value.setRGB(cur.horizon.x, cur.horizon.y, cur.horizon.z, THREE.SRGBColorSpace)
  })
  return (
    <mesh rotation-x={-Math.PI / 2} receiveShadow material={mat}>
      <circleGeometry args={[GROUND_R, 160]} />
    </mesh>
  )
}

function makeBuildingGeometry() {
  const g = new THREE.BoxGeometry(1, 1, 1)
  g.translate(0, 0.5, 0)
  const pos = g.getAttribute('position')
  const nor = g.getAttribute('normal')
  const col = new Float32Array(pos.count * 3)
  for (let i = 0; i < pos.count; i++) {
    const roof = nor.getY(i) > 0.5
    const y = pos.getY(i)
    const c = roof ? [0.2, 0.235, 0.27] : y > 0.5 ? [0.135, 0.16, 0.19] : [0.06, 0.07, 0.085]
    col.set(c, i * 3)
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3))
  g.setAttribute('edgeUv', g.getAttribute('uv').clone())
  return g
}

function Buildings({ city, stationKey }: { city: CityLayout; stationKey: number }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const group = useRef<THREE.Group>(null)
  const grow = useRef(0)
  const geo = useMemo(makeBuildingGeometry, [])
  const mat = useMemo(() => {
    const m = new THREE.MeshLambertMaterial({ vertexColors: true })
    m.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nattribute vec2 edgeUv;\nvarying vec2 vEdgeUv;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvEdgeUv = edgeUv;')
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vEdgeUv;')
        .replace(
          '#include <opaque_fragment>',
          `vec2 eu = min(vEdgeUv, 1.0 - vEdgeUv);
vec2 ew = fwidth(vEdgeUv) * 1.2;
float edge = 1.0 - min(smoothstep(0.0, ew.x, eu.x), smoothstep(0.0, ew.y, eu.y));
outgoingLight += vec3(0.035) * edge;
#include <opaque_fragment>`,
        )
    }
    m.customProgramCacheKey = () => 'column-buildings'
    return m
  }, [])
  useEffect(
    () => () => {
      geo.dispose()
      mat.dispose()
    },
    [geo, mat],
  )
  useLayoutEffect(() => {
    const m = mesh.current
    if (!m) return
    m.instanceMatrix.array.set(city.buildings.matrices)
    m.instanceMatrix.needsUpdate = true
    const c = new THREE.Color()
    for (let i = 0; i < city.buildings.count; i++) {
      const t = city.buildings.tones[i]
      m.setColorAt(i, c.setRGB(t, t, t * 1.02))
    }
    if (m.instanceColor) m.instanceColor.needsUpdate = true
    m.computeBoundingSphere()
    grow.current = 0
  }, [city, stationKey])
  useFrame((_, dt) => {
    if (!group.current || grow.current >= 1) return
    grow.current = Math.min(1, grow.current + dt * 1.4)
    const e = 1 - (1 - grow.current) ** 3
    group.current.scale.y = Math.max(0.001, e)
  })
  return (
    <group ref={group}>
      <instancedMesh
        key={city.buildings.count}
        ref={mesh}
        args={[geo, mat, city.buildings.count]}
        castShadow
        receiveShadow
      />
    </group>
  )
}

function Trees({ city }: { city: CityLayout }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const geo = useMemo(() => new THREE.IcosahedronGeometry(1, 0), [])
  const mat = useMemo(() => new THREE.MeshLambertMaterial({ color: '#22362b', flatShading: true }), [])
  useEffect(
    () => () => {
      geo.dispose()
      mat.dispose()
    },
    [geo, mat],
  )
  useLayoutEffect(() => {
    const m = mesh.current
    if (!m) return
    m.instanceMatrix.array.set(city.trees.matrices)
    m.instanceMatrix.needsUpdate = true
    const c = new THREE.Color()
    for (let i = 0; i < city.trees.count; i++) {
      const t = city.trees.tones[i]
      m.setColorAt(i, c.setRGB(t, t, t))
    }
    if (m.instanceColor) m.instanceColor.needsUpdate = true
    m.computeBoundingSphere()
  }, [city])
  if (city.trees.count === 0) return null
  return <instancedMesh key={city.trees.count} ref={mesh} args={[geo, mat, city.trees.count]} castShadow receiveShadow />
}

function useShader(make: () => THREE.ShaderMaterial) {
  const mat = useMemo(make, [])
  useEffect(() => () => mat.dispose(), [mat])
  return mat
}

function Column({ cur }: { cur: Cur }) {
  const shellG = useRef<THREE.Mesh>(null)
  const hazeG = useRef<THREE.Mesh>(null)
  const lid = useRef<THREE.Group>(null)
  const geo = useMemo(() => {
    const g = new THREE.CylinderGeometry(COL_R, COL_R, 1, 128, 1, true)
    g.translate(0, 0.5, 0)
    return g
  }, [])
  const hazeGeo = useMemo(() => {
    const g = new THREE.CylinderGeometry(COL_R * 0.985, COL_R * 0.985, 1, 96, 1, true)
    g.translate(0, 0.5, 0)
    return g
  }, [])
  useEffect(
    () => () => {
      geo.dispose()
      hazeGeo.dispose()
    },
    [geo, hazeGeo],
  )
  const shell = useShader(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { uColor: { value: PAPER.clone() }, uOpacity: { value: 1 } },
        vertexShader: COLUMN_VERT,
        fragmentShader: SHELL_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
  )
  const haze = useShader(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Vector3() }, uDensity: { value: 0.1 } },
        vertexShader: COLUMN_VERT,
        fragmentShader: HAZE_FRAG,
        transparent: true,
        depthWrite: false,
      }),
  )
  const lidMat = useShader(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uR: { value: COL_R },
          uOuter: { value: 14 },
          uColor: { value: PAPER.clone() },
          uOpacity: { value: 1 },
        },
        vertexShader: DISC_VERT,
        fragmentShader: LID_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
  )
  const foot = useShader(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Vector3() }, uOpacity: { value: 0.5 }, uR: { value: COL_R } },
        vertexShader: DISC_VERT,
        fragmentShader: FOOT_FRAG,
        transparent: true,
        depthWrite: false,
      }),
  )
  const ringMat = useMemo(() => new THREE.MeshBasicMaterial({ color: '#ECE7DC', transparent: true, opacity: 0.85, fog: false }), [])
  const footRingMat = useMemo(() => new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, fog: false }), [])
  useEffect(
    () => () => {
      ringMat.dispose()
      footRingMat.dispose()
    },
    [ringMat, footRingMat],
  )
  useFrame(() => {
    shellG.current?.scale.set(1, cur.lid, 1)
    hazeG.current?.scale.set(1, cur.lid, 1)
    lid.current?.position.set(0, cur.lid, 0)
    haze.uniforms.uColor.value.copy(cur.rgb)
    // Haze opacity rises with concentration (CPCB 'Very poor' starts at 120 µg/m³).
    haze.uniforms.uDensity.value = 0.035 + 0.16 * smooth(0, 200, cur.pm)
    lidMat.uniforms.uTime.value = cur.time
    foot.uniforms.uColor.value.copy(cur.rgb)
    foot.uniforms.uOpacity.value = 0.25 + 0.35 * smooth(0, 160, cur.pm)
    footRingMat.color.setRGB(cur.rgb.x, cur.rgb.y, cur.rgb.z, THREE.SRGBColorSpace)
  })
  return (
    <group>
      <mesh ref={shellG} geometry={geo} material={shell} renderOrder={3} />
      <mesh ref={hazeG} geometry={hazeGeo} material={haze} renderOrder={2} />
      <mesh rotation-x={-Math.PI / 2} position-y={0.03} material={foot} renderOrder={1}>
        <circleGeometry args={[COL_R, 96]} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={0.035} material={footRingMat} renderOrder={1}>
        <ringGeometry args={[COL_R - 0.05, COL_R + 0.05, 160]} />
      </mesh>
      <group ref={lid}>
        <mesh rotation-x={-Math.PI / 2} material={lidMat} renderOrder={4}>
          <circleGeometry args={[14, 128]} />
        </mesh>
        <mesh rotation-x={-Math.PI / 2} material={ringMat} renderOrder={5}>
          <ringGeometry args={[COL_R - 0.045, COL_R + 0.045, 192]} />
        </mesh>
      </group>
    </group>
  )
}

function Particles({ cur, clean = false }: { cur: Cur; clean?: boolean }) {
  const n = clean ? N_CLEAN : N_PARTICLES
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    const rand = new Float32Array(n * 4)
    const meta = new Float32Array(n * 2)
    let s = clean ? 97 : 13
    const r = () => {
      s = (s * 16807) % 2147483647
      return s / 2147483647
    }
    for (let i = 0; i < n; i++) {
      rand.set([r() * 2 - 1, r() * 2 - 1, r(), r()], i * 4)
      meta.set([r(), 0.55 + r() * 0.9], i * 2)
    }
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
    g.setAttribute('aRand', new THREE.BufferAttribute(rand, 4))
    g.setAttribute('aMeta', new THREE.BufferAttribute(meta, 2))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 16, 0), 40)
    return g
  }, [n, clean])
  useEffect(() => () => geo.dispose(), [geo])
  const dpr = useThree((s) => s.viewport.dpr)
  const mat = useShader(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uBase: { value: 0.05 },
          uTop: { value: 1 },
          uFrac: { value: 1 },
          uSize: { value: clean ? 3.2 : 4.4 },
          uPx: { value: 1 },
          uMixPhase: { value: 0 },
          uStrat: { value: 1 },
          uR: { value: clean ? COL_R + 3 : COL_R },
          uWobble: { value: 0.2 },
          uDrift: { value: new THREE.Vector2() },
          uColor: { value: clean ? PAPER.clone() : new THREE.Vector3() },
          uOpacity: { value: clean ? 0.22 : 0.78 },
        },
        vertexShader: PARTICLE_VERT,
        fragmentShader: PARTICLE_FRAG,
        transparent: true,
        depthWrite: false,
      }),
  )
  useFrame(() => {
    const u = mat.uniforms
    u.uTime.value = cur.time
    u.uPx.value = dpr
    u.uDrift.value.copy(cur.drift)
    u.uMixPhase.value = cur.mixPhase
    if (clean) {
      u.uBase.value = cur.lid + 0.6
      u.uTop.value = Math.max(cur.lid + 4, TOP_U)
      u.uStrat.value = 1
      u.uWobble.value = 0.3
    } else {
      u.uTop.value = cur.lid
      // Stable nights keep aerosol bottom-heavy; daytime convection mixes it evenly.
      u.uStrat.value = 1.0 + 0.9 * (1 - cur.conv)
      u.uWobble.value = 0.12 + 0.3 * cur.conv
      // More particles as PM2.5 rises; the full set is reached at 150 µg/m³.
      u.uFrac.value = 0.12 + 0.88 * Math.min(1, cur.pm / 150)
      u.uColor.value.copy(cur.rgb)
    }
  })
  return <points geometry={geo} material={mat} frustumCulled={false} renderOrder={6} />
}

function WindStreaks({ cur }: { cur: Cur }) {
  const geo = useMemo(() => {
    const g = new THREE.InstancedBufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3))
    g.setAttribute('corner', new THREE.BufferAttribute(new Float32Array([0, -1, 1, -1, 1, 1, 0, 1]), 2))
    g.setIndex([0, 1, 2, 0, 2, 3])
    const levels = [0.18, 0.5, 0.84]
    const perLevel = 11
    const inst = new Float32Array(levels.length * perLevel * 3)
    let k = 0
    let s = 7
    const r = () => {
      s = (s * 16807) % 2147483647
      return s / 2147483647
    }
    for (const h of levels)
      for (let i = 0; i < perLevel; i++) {
        const lane = -0.85 + (1.7 * (i + 0.5)) / perLevel + (r() - 0.5) * 0.08
        inst.set([lane, h + (r() - 0.5) * 0.06, r()], k * 3)
        k++
      }
    g.setAttribute('aInst', new THREE.InstancedBufferAttribute(inst, 3))
    g.instanceCount = k
    return g
  }, [])
  useEffect(() => () => geo.dispose(), [geo])
  const mat = useShader(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uDir: { value: new THREE.Vector2(1, 0) },
          uTravel: { value: 0 },
          uLen: { value: 2 },
          uLid: { value: 1 },
          uR: { value: COL_R + 2.2 },
          uWidth: { value: 0.045 },
          uColor: { value: WIND.clone() },
          uOpacity: { value: 0.7 },
        },
        vertexShader: STREAK_VERT,
        fragmentShader: STREAK_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
  )
  useFrame(() => {
    const sp = Math.hypot(cur.u, cur.v)
    const u = mat.uniforms
    if (sp > 0.02) u.uDir.value.set(cur.u / sp, -cur.v / sp)
    u.uTravel.value = cur.travel
    u.uLen.value = 0.6 + Math.min(4.2, sp * 1.1)
    u.uLid.value = cur.lid
    u.uOpacity.value = 0.22 + 0.6 * smooth(0, 3.5, sp)
  })
  return <mesh geometry={geo} material={mat} frustumCulled={false} renderOrder={7} />
}

function SourcePlumes({ cur, sources }: { cur: Cur; sources: NearbySource[] }) {
  const group = useRef<THREE.Group>(null)
  const geometry = useMemo(() => new THREE.CylinderGeometry(1, 0.035, 1, 32, 1, true), [])
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#CFE3F2',
        transparent: true,
        opacity: 0.09,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    [],
  )
  const placements = useMemo(
    () =>
      sources.map((source) => {
        const [bx, bz] = bearingVec(source.bearing)
        const distance = (source.km * 1000) / M_PER_UNIT
        return { start: new THREE.Vector3(bx * distance, 0.4, bz * distance) }
      }),
    [sources],
  )
  const direction = useMemo(() => new THREE.Vector3(), [])
  const centre = useMemo(() => new THREE.Vector3(), [])
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )
  useFrame(() => {
    if (!group.current) return
    const speed = Math.hypot(cur.u, cur.v)
    const wx = speed > 0.02 ? cur.u / speed : 1
    const wz = speed > 0.02 ? -cur.v / speed : 0
    const length = 2.8 + Math.min(4.8, speed * 1.25)
    // Reserve room for the tilted plume rim as well as its centreline below the lid.
    const radius = Math.max(0.02, Math.min(0.34 + length * 0.09, cur.lid - 0.45))
    const rise = Math.max(0, Math.min(cur.lid - 0.45 - radius, length * 0.18))
    direction.set(wx, rise / length, wz).normalize()
    material.opacity = 0.045 + 0.065 * smooth(0, 3, speed)
    group.current.visible = sources.length > 0 && speed > 0.02 && cur.lid > 0.45
    for (let i = 0; i < group.current.children.length; i++) {
      const mesh = group.current.children[i]
      const source = placements[i]
      if (!source) continue
      centre.copy(source.start).addScaledVector(direction, length * 0.5)
      mesh.position.copy(centre)
      mesh.quaternion.setFromUnitVectors(UP, direction)
      mesh.scale.set(radius, length, radius)
    }
  })
  return (
    <group ref={group}>
      {placements.map((_, i) => (
        <mesh key={`${sources[i].name}-${i}`} geometry={geometry} material={material} renderOrder={2} />
      ))}
    </group>
  )
}

const labelShadow = { textShadow: '0 1px 2px rgba(5,9,14,0.85), 0 0 8px rgba(5,9,14,0.6)' }

function Label({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={`pointer-events-none select-none whitespace-nowrap ${className ?? ''}`} style={labelShadow}>
      {children}
    </div>
  )
}

function ConcentrationProfile({ cur }: { cur: Cur }) {
  const profile = useRef<Line2 | LineSegments2>(null)
  const initialPoints = useMemo<[number, number, number][]>(
    () =>
      Array.from({ length: PROFILE_POINTS }, (_, i) => [
        -0.55,
        i < PROFILE_GRID_POINTS ? (i / (PROFILE_GRID_POINTS - 1)) * PROFILE_TOP_U : 0,
        0,
      ]),
    [],
  )
  const initialColors = useMemo<[number, number, number][]>(
    () => Array.from({ length: PROFILE_POINTS }, () => [WIND.x, WIND.y, WIND.z]),
    [],
  )
  // Reused frame buffers keep the Drei line geometry and its GPU allocations stable.
  const xz = useMemo(() => new Float32Array(PROFILE_POINTS * 2), [])
  const rgb = useMemo(() => new Float32Array(PROFILE_POINTS * 3), [])
  const heights = useMemo(() => new Float32Array(PROFILE_POINTS), [])
  useFrame(() => {
    const line = profile.current
    if (!line) return
    const stratification = 1 + 0.9 * (1 - cur.conv)
    const c0 = Math.max(0, cur.pm)
    const lid = Math.max(0, Math.min(PROFILE_TOP_U, cur.lid))
    for (let i = 0; i < PROFILE_GRID_POINTS; i++) {
      heights[i] = (i / (PROFILE_GRID_POINTS - 1)) * PROFILE_TOP_U
    }
    heights[PROFILE_GRID_POINTS] = Math.max(0, lid - PROFILE_STEP_HALF_WIDTH_U)
    heights[PROFILE_GRID_POINTS + 1] = Math.min(PROFILE_TOP_U, lid + PROFILE_STEP_HALF_WIDTH_U)
    heights.sort()
    for (let i = 0; i < PROFILE_POINTS; i++) {
      const z = heights[i]
      const fraction = verticalProfileFraction(z * M_PER_UNIT, cur.lid * M_PER_UNIT, stratification)
      const concentration = c0 * fraction
      const color = z <= lid ? cur.rgb : WIND
      xz[i * 2] = -0.55 - 2.5 * (concentration / Math.max(1, c0))
      xz[i * 2 + 1] = z
      rgb[i * 3] = color.x
      rgb[i * 3 + 1] = color.y
      rgb[i * 3 + 2] = color.z
    }
    const start = line.geometry.getAttribute('instanceStart')
    const end = line.geometry.getAttribute('instanceEnd')
    const colorStart = line.geometry.getAttribute('instanceColorStart')
    const colorEnd = line.geometry.getAttribute('instanceColorEnd')
    for (let i = 0; i < PROFILE_POINTS - 1; i++) {
      start.setXYZ(i, xz[i * 2], xz[i * 2 + 1], 0)
      end.setXYZ(i, xz[i * 2 + 2], xz[i * 2 + 3], 0)
      colorStart.setXYZ(i, rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2])
      colorEnd.setXYZ(i, rgb[i * 3 + 3], rgb[i * 3 + 4], rgb[i * 3 + 5])
    }
    start.needsUpdate = true
    colorStart.needsUpdate = true
  })
  return (
    <>
      <Line
        ref={profile}
        points={initialPoints}
        vertexColors={initialColors}
        color="#FFFFFF"
        lineWidth={1.8}
        transparent
        opacity={0.95}
        frustumCulled={false}
      />
      <Html portal={HTML_PORTAL} position={[-3.1, PROFILE_TOP_U - 0.5, 0]} center zIndexRange={[5, 0]} pointerEvents="none">
        <Label className="font-mono text-[9px] text-paper/60">PM2.5 C(z)</Label>
      </Html>
    </>
  )
}

/** Height axis + lid labels, kept on the camera's left/right side of the column. */
function Annotations({ cur, lidText, peakText }: { cur: Cur; lidText: string; peakText: string | null }) {
  const axis = useRef<THREE.Group>(null)
  const lidLabel = useRef<THREE.Group>(null)
  const peakLabel = useRef<THREE.Group>(null)
  const peakRing = useRef<THREE.Group>(null)
  const right = useMemo(() => new THREE.Vector3(), [])
  const camera = useThree((s) => s.camera)
  const axisGeo = useMemo(() => {
    const top = AXIS_TICKS_M[AXIS_TICKS_M.length - 1] / M_PER_UNIT
    const pts: number[] = [0, 0, 0, 0, top, 0]
    for (const m of AXIS_TICKS_M) pts.push(0, m / M_PER_UNIT, 0, -0.45, m / M_PER_UNIT, 0)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
    return g
  }, [])
  const axisMat = useMemo(() => new THREE.LineBasicMaterial({ color: '#ECE7DC', transparent: true, opacity: 0.45, fog: false }), [])
  useEffect(
    () => () => {
      axisGeo.dispose()
      axisMat.dispose()
    },
    [axisGeo, axisMat],
  )
  const circle = useMemo(() => {
    const pts: [number, number, number][] = []
    for (let i = 0; i <= 160; i++) {
      const a = (i / 160) * Math.PI * 2
      pts.push([Math.cos(a) * COL_R, 0, Math.sin(a) * COL_R])
    }
    return pts
  }, [])
  useFrame(() => {
    right.setFromMatrixColumn(camera.matrixWorld, 0)
    right.y = 0
    right.normalize()
    const a = axis.current
    if (a) {
      a.position.set(-right.x * (COL_R + 1.6), 0, -right.z * (COL_R + 1.6))
      a.rotation.y = Math.atan2(-right.z, right.x)
    }
    lidLabel.current?.position.set(right.x * (COL_R + 0.35), cur.lid, right.z * (COL_R + 0.35))
    const pk = peakRing.current
    if (pk) {
      const show = cur.peak > 0 && cur.peak - cur.lid > 1.2
      pk.visible = show
      pk.position.y = Math.max(cur.peak, 0)
    }
    peakLabel.current?.position.set(right.x * (COL_R + 0.35), 0, right.z * (COL_R + 0.35))
  })
  return (
    <>
      <group ref={axis}>
        <lineSegments geometry={axisGeo} material={axisMat} />
        <ConcentrationProfile cur={cur} />
        {AXIS_TICKS_M.map((m) => (
          <Html portal={HTML_PORTAL} key={m} position={[-0.6, m / M_PER_UNIT, 0]} zIndexRange={[5, 0]} pointerEvents="none" style={{ transform: 'translate(-100%, -50%)' }}>
            <Label className="font-mono text-[10px] tnum text-paper/70">{m === 0 ? 'Ground' : `${fmtInt(m)} m`}</Label>
          </Html>
        ))}
      </group>
      <group ref={lidLabel}>
        <Html portal={HTML_PORTAL} zIndexRange={[6, 0]} pointerEvents="none" style={{ transform: 'translate(10px, -50%)' }}>
          <Label className="flex items-center gap-2">
            <span className="h-px w-5 bg-paper/70" />
            <span className="text-[12px] font-medium text-paper">{lidText}</span>
          </Label>
        </Html>
      </group>
      <group ref={peakRing}>
        <Line points={circle} color="#ECE7DC" lineWidth={1} dashed dashSize={0.35} gapSize={0.3} transparent opacity={0.4} />
        <group ref={peakLabel}>
          {peakText && (
            <Html portal={HTML_PORTAL} zIndexRange={[5, 0]} pointerEvents="none" style={{ transform: 'translate(10px, -50%)' }}>
              <Label className="text-[11px] text-paper/60">{peakText}</Label>
            </Html>
          )}
        </group>
      </group>
    </>
  )
}

function GroundMarks({ city, sources }: { city: CityLayout; sources: NearbySource[] }) {
  const pinMat = useMemo(() => new THREE.MeshBasicMaterial({ color: '#ECE7DC', fog: false }), [])
  const srcMat = useMemo(() => new THREE.MeshBasicMaterial({ color: '#ECE7DC', transparent: true, opacity: 0.55 }), [])
  useEffect(
    () => () => {
      pinMat.dispose()
      srcMat.dispose()
    },
    [pinMat, srcMat],
  )
  const north: [number, number, number] = [0, 0.05, -(TILE_R + 1.1)]
  const seaPos = useMemo((): [number, number, number] | null => {
    if (city.coastDist == null) return null
    const d = Math.min(GROUND_R * 0.72, city.coastDist + 4)
    return [city.coastDir[0] * d, 0.1, city.coastDir[1] * d]
  }, [city])
  return (
    <>
      {/* The monitor itself, at the centre of the column. */}
      <mesh position={[0, 0.3, 0]} material={pinMat}>
        <cylinderGeometry args={[0.025, 0.025, 0.6, 8]} />
      </mesh>
      <mesh position={[0, 0.62, 0]} material={pinMat}>
        <sphereGeometry args={[0.08, 16, 12]} />
      </mesh>
      <Html portal={HTML_PORTAL} position={north} center zIndexRange={[5, 0]} pointerEvents="none">
        <Label className="font-mono text-[11px] font-medium text-paper/80">N</Label>
      </Html>
      <mesh position={[0, 0.04, -(TILE_R + 0.35)]} rotation-x={-Math.PI / 2} material={pinMat}>
        <circleGeometry args={[0.12, 3]} />
      </mesh>
      <Html portal={HTML_PORTAL} position={[0, 0.05, TILE_R + 0.9]} center zIndexRange={[5, 0]} pointerEvents="none">
        <Label className="font-mono text-[10px] text-paper/50">1 km</Label>
      </Html>
      {seaPos && (
        <Html portal={HTML_PORTAL} position={seaPos} center zIndexRange={[5, 0]} pointerEvents="none">
          <Label className="text-[11px] italic text-paper/55">Arabian Sea</Label>
        </Html>
      )}
      {sources.map((s) => {
        const [bx, bz] = bearingVec(s.bearing)
        const d = (s.km * 1000) / M_PER_UNIT
        return (
          <group key={s.name} position={[bx * d, 0, bz * d]}>
            <mesh rotation-x={-Math.PI / 2} position-y={0.05} material={srcMat}>
              <ringGeometry args={[0.32, 0.4, 48]} />
            </mesh>
            <mesh position-y={0.9} material={srcMat}>
              <cylinderGeometry args={[0.015, 0.015, 1.8, 6]} />
            </mesh>
            <Html portal={HTML_PORTAL} position={[0, 1.9, 0]} center zIndexRange={[5, 0]} pointerEvents="none">
              <Label className="text-center text-[11px] leading-tight text-paper/75">
                {s.name}
                <span className="block font-mono text-[10px] text-paper/45">{s.km.toFixed(1)} km</span>
              </Label>
            </Html>
          </group>
        )
      })}
    </>
  )
}

function FluxVectors({ stationId, fluxes, stations }: { stationId: number; fluxes: FluxEdge[]; stations: Station[] }) {
  const headGeometry = useMemo(() => new THREE.ConeGeometry(0.18, 0.52, 12), [])
  const headMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ color: '#CFE3F2', transparent: true, opacity: 0.78, depthWrite: false }),
    [],
  )
  useEffect(
    () => () => {
      headGeometry.dispose()
      headMaterial.dispose()
    },
    [headGeometry, headMaterial],
  )
  const ring = useMemo<[number, number, number][]>(
    () =>
      Array.from({ length: 129 }, (_, i) => {
        const angle = (i / 128) * Math.PI * 2
        return [Math.sin(angle) * TILE_R, 0.08, -Math.cos(angle) * TILE_R]
      }),
    [],
  )
  const vectors = useMemo(() => {
    const selected = stations.find((station) => station.id === stationId)
    if (!selected) return []
    return fluxes.flatMap((edge) => {
      const incoming = edge.dst === stationId
      if (!incoming && edge.src !== stationId) return []
      const otherId = incoming ? edge.src : edge.dst
      const other = stations.find((station) => station.id === otherId)
      if (!other) return []
      const east = (other.lon - selected.lon) * Math.cos(((other.lat + selected.lat) * Math.PI) / 360)
      const north = other.lat - selected.lat
      const bearing = (Math.atan2(east, north) * 180) / Math.PI
      const [bx, bz] = bearingVec(bearing)
      const outer: [number, number, number] = [bx * TILE_R, 0.16, bz * TILE_R]
      const inner: [number, number, number] = [bx * (COL_R + 0.9), 0.16, bz * (COL_R + 0.9)]
      const start = incoming ? outer : inner
      const end = incoming ? inner : outer
      const dx = end[0] - start[0]
      const dz = end[2] - start[2]
      const length = Math.hypot(dx, dz)
      const direction = new THREE.Vector3(dx / length, 0, dz / length)
      const headScale = 0.8 + Math.min(0.8, edge.flux / 150)
      const headLength = 0.52 * headScale
      return [
        {
          key: `${edge.src}-${edge.dst}`,
          start,
          end,
          head: [end[0] - direction.x * headLength * 0.5, end[1], end[2] - direction.z * headLength * 0.5] as [
            number,
            number,
            number,
          ],
          quaternion: new THREE.Quaternion().setFromUnitVectors(UP, direction),
          headScale,
          lineWidth: 1.2 + Math.min(3, edge.flux / 25),
          opacity: 0.4 + Math.min(0.5, edge.flux / 100),
        },
      ]
    })
  }, [stationId, fluxes, stations])
  return (
    <group>
      <Line points={ring} color="#CFE3F2" lineWidth={0.7} transparent opacity={0.2} />
      {vectors.map((vector) => (
        <group key={vector.key}>
          <Line
            points={[vector.start, vector.end]}
            color="#CFE3F2"
            lineWidth={vector.lineWidth}
            transparent
            opacity={vector.opacity}
            depthWrite={false}
          />
          <mesh
            geometry={headGeometry}
            material={headMaterial}
            position={vector.head}
            quaternion={vector.quaternion}
            scale={vector.headScale}
            renderOrder={8}
          />
        </group>
      ))}
    </group>
  )
}

function Rig({ cur }: { cur: Cur }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const resume = useRef<number | undefined>(undefined)
  const reduced = useMemo(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false, [])
  const camera = useThree((s) => s.camera)
  useEffect(() => () => window.clearTimeout(resume.current), [])
  useFrame((_, rawDt) => {
    const c = controls.current
    if (!c) return
    const dt = Math.min(rawDt, 0.1)
    // Frame the whole day's range: target follows the larger of the lid and today's peak.
    const span = Math.max(cur.lid, cur.peak)
    const want = Math.max(3, Math.min(13, span * 0.42))
    const dy = (want - c.target.y) * (1 - Math.exp(-dt * 1.2))
    c.target.y += dy
    camera.position.y += dy
  })
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      target={[0, 8, 0]}
      enableDamping
      dampingFactor={0.07}
      enablePan={false}
      minDistance={14}
      maxDistance={105}
      minPolarAngle={0.2}
      maxPolarAngle={1.47}
      autoRotate={!reduced}
      autoRotateSpeed={0.35}
      rotateSpeed={0.6}
      zoomSpeed={0.8}
      onStart={() => {
        window.clearTimeout(resume.current)
        if (controls.current) controls.current.autoRotate = false
      }}
      onEnd={() => {
        window.clearTimeout(resume.current)
        resume.current = window.setTimeout(() => {
          if (controls.current && !reduced) controls.current.autoRotate = true
        }, 5000)
      }}
    />
  )
}

/** Shifts the projection so the column sits in the free area right of the HUD and above the timeline. */
function FrameOffset({ x, y }: { x: number; y: number }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera
  const size = useThree((s) => s.size)
  useLayoutEffect(() => {
    camera.setViewOffset(size.width, size.height, -x, y, size.width, size.height)
    camera.updateProjectionMatrix()
    return () => {
      camera.clearViewOffset()
      camera.updateProjectionMatrix()
    }
  }, [camera, size.width, size.height, x, y])
  return null
}

export function ColumnScene({
  targets,
  cityInput,
  stationId,
  stations,
  sources,
  fluxes,
  lidText,
  peakText,
}: {
  targets: RefObject<SceneTargets>
  cityInput: CityInput
  stationId: number
  stations: Station[]
  sources: NearbySource[]
  fluxes: FluxEdge[]
  lidText: string
  peakText: string | null
}) {
  const cur = useMemo(makeCur, [])
  const { seed, density, roughness, green, coastKm } = cityInput
  const city = useMemo(() => buildCity({ seed, density, roughness, green, coastKm }), [seed, density, roughness, green, coastKm])
  useEffect(() => () => city.greenTex.dispose(), [city])
  const width = useThree((s) => s.size.width)
  const offsetX = width > 900 ? Math.min(190, width * 0.13) : 0
  return (
    <>
      <FrameOffset x={offsetX} y={52} />
      <Driver targets={targets} cur={cur} />
      <Sky cur={cur} />
      <Lights cur={cur} />
      <Ground city={city} cur={cur} />
      <Buildings city={city} stationKey={stationId} />
      <Trees city={city} />
      <GroundMarks city={city} sources={sources} />
      <SourcePlumes cur={cur} sources={sources} />
      <FluxVectors stationId={stationId} fluxes={fluxes} stations={stations} />
      <Column cur={cur} />
      <Particles cur={cur} />
      <Particles cur={cur} clean />
      <WindStreaks cur={cur} />
      <Annotations cur={cur} lidText={lidText} peakText={peakText} />
      <Rig cur={cur} />
    </>
  )
}
