import { useMemo, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { useData, useTwin } from '../../state/store'
import { fluxEdges } from '../../lib/physics'
import { ColumnHud } from './ColumnHud'
import { deriveColumn, fmtInt, sceneTargets, type SceneTargets } from './derive'
import { ColumnScene } from './Scene'

export function ColumnView() {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const stationId = useTwin((s) => s.stationId)
  const st = useMemo(() => deriveColumn(data, stationId, hour), [data, stationId, hour])
  const fluxes = useMemo(
    () => fluxEdges(data, hour).filter((edge) => edge.src === stationId || edge.dst === stationId),
    [data, hour, stationId],
  )
  const targets = useRef<SceneTargets>(sceneTargets(st))
  targets.current = sceneTargets(st)

  const h = st.met.pblh
  const lidText = h == null ? 'Mixing height · no data' : h < 50 ? 'Mixing height · under 50 m' : `Mixing height · ${fmtInt(h)} m`
  const peakText = st.dayPeak && h != null && st.dayPeak.value > h * 1.15 ? `Today's peak · ${fmtInt(st.dayPeak.value)} m at ${st.dayPeak.time}` : null

  return (
    <div className="absolute inset-0">
      <div className="absolute inset-0 isolate">
        <Canvas
          dpr={[1, 2]}
          flat
          shadows="percentage"
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          camera={{ fov: 34, near: 0.5, far: 1500, position: [44, 26, 26] }}
        >
          <ColumnScene
            targets={targets}
            cityInput={{ seed: stationId, ...st.morphology }}
            stationId={stationId}
            stations={data.stations.stations}
            sources={st.sources}
            fluxes={fluxes}
            lidText={lidText}
            peakText={peakText}
          />
        </Canvas>
      </div>
      <ColumnHud data={data} st={st} stationId={stationId} fluxes={fluxes} />
      <div className="pointer-events-none absolute right-5 top-20 z-10 text-right text-2xs text-faint">
        Drag to orbit · scroll to zoom
      </div>
    </div>
  )
}
