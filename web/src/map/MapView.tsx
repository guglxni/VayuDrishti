import '../lib/maplibreWorker'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Map as MlMap, NavigationControl, type GeoJSONSource, type ImageSource } from 'maplibre-gl'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { ArcLayer, ColumnLayer, PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import { TripsLayer } from '@deck.gl/geo-layers'
import { PathStyleExtension } from '@deck.gl/extensions'
import type { Layer, PickingInfo } from '@deck.gl/core'
import { useData, useTwin } from '../state/store'
import type { TwinData } from '../data/types'
import { pmColor, pmRGB } from '../lib/aqi'
import { fluxEdges, type FluxEdge } from '../lib/physics'
import { sampleAt } from '../lib/series'
import { stationName, TARGET_IDS } from '../lib/stations'
import { BEFORE_DATA_LAYER, BEFORE_LABELS, buildBasemapStyle } from './basemap'
import { renderSurface, stationValues, SURFACE_COORDS, traceStreamlines, WIND_LOOP, type Streamline } from './fields'
import { StationCard } from './StationCard'
import { LayerPanel } from './LayerPanel'
import { WardPopover, type WardPick } from './WardPopover'

/** Vertical exaggeration for station columns: metres of column per µg/m³. */
const ELEV_PER_UG = 60

export const CAMERAS = {
  sea: { center: [72.872, 19.085] as [number, number], zoom: 10.85, pitch: 58, bearing: 96 },
  north: { center: [72.875, 19.09] as [number, number], zoom: 10.55, pitch: 40, bearing: -12 },
  deonar: { center: [72.923, 19.051] as [number, number], zoom: 13.1, pitch: 62, bearing: -75 },
  malad: { center: [72.84, 19.187] as [number, number], zoom: 13, pitch: 58, bearing: 65 },
  borivali: { center: [72.868, 19.229] as [number, number], zoom: 13, pitch: 58, bearing: 45 },
  byculla: { center: [72.833, 18.976] as [number, number], zoom: 13.2, pitch: 60, bearing: 35 },
}

export type Cam = keyof typeof CAMERAS

const TARGET_SET = new Set<number>(TARGET_IDS)

interface FluxArc extends FluxEdge {
  kind: 'airphynet'
  srcName: string
  dstName: string
  sourcePosition: [number, number, number]
  targetPosition: [number, number, number]
}

function wardGeoJSON(data: TwinData, hour: number) {
  return {
    type: 'FeatureCollection' as const,
    features: data.wards.wards.map((w, id) => ({
      type: 'Feature' as const,
      id,
      properties: { code: w.code, name: w.name, pm: sampleAt(w.series, hour), color: pmColor(sampleAt(w.series, hour)) },
      geometry: { type: 'Polygon' as const, coordinates: [[...w.polygon, w.polygon[0]]] },
    })),
  }
}

function fontReady() {
  return Promise.all([
    document.fonts.load('600 32px "IBM Plex Sans Condensed"'),
    document.fonts.load('500 32px "IBM Plex Sans Condensed"'),
  ]).catch(() => undefined)
}

export function MapView() {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const stationId = useTwin((s) => s.stationId)
  const setStation = useTwin((s) => s.setStation)
  const setWard = useTwin((s) => s.setWard)
  const layers = useTwin((s) => s.layers)

  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MlMap | null>(null)
  const overlayRef = useRef<MapboxOverlay | null>(null)
  const surfaceCanvas = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const [ready, setReady] = useState(false)
  const [fontsLoaded, setFontsLoaded] = useState(false)
  const [mapError, setMapError] = useState<string | null>(null)
  const [wardPick, setWardPick] = useState<WardPick | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number; content: ReactNode } | null>(null)

  useEffect(() => {
    void fontReady().then(() => setFontsLoaded(true))
  }, [])

  // ----- Map + overlay lifecycle -----
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let map: MlMap
    try {
      map = new MlMap({
        container,
        style: buildBasemapStyle(),
        ...CAMERAS.sea,
        maxPitch: 75,
        minZoom: 9,
        maxZoom: 17,
        maxBounds: [
          [72.3, 18.6],
          [73.45, 19.6],
        ],
        attributionControl: { compact: true },
        canvasContextAttributes: { antialias: true },
      })
    } catch (e) {
      setMapError(e instanceof Error ? e.message : String(e))
      return
    }
    mapRef.current = map
    const resizeObserver = new ResizeObserver(() => mapRef.current?.resize())
    resizeObserver.observe(container)
    map.addControl(new NavigationControl({ visualizePitch: true }), 'bottom-right')
    map.on('error', (ev) => {
      const msg = ev.error?.message ?? ''
      if (/webgl|worker/i.test(msg)) setMapError(msg)
    })

    const overlay = new MapboxOverlay({ interleaved: false, layers: [] })
    overlayRef.current = overlay
    map.addControl(overlay)

    map.on('load', () => {
      const wardsGeo = wardGeoJSON(data, useTwin.getState().hour)
      const wardLabels = {
        type: 'FeatureCollection' as const,
        features: data.wards.wards.map((w) => ({
          type: 'Feature' as const,
          properties: { code: w.code },
          geometry: { type: 'Point' as const, coordinates: [w.lon, w.lat] },
        })),
      }
      renderSurface(data, useTwin.getState().hour, surfaceCanvas.current)
      map.addSource('pm-surface', {
        type: 'image',
        url: surfaceCanvas.current.toDataURL(),
        coordinates: SURFACE_COORDS,
      })
      map.addLayer(
        {
          id: 'pm-surface',
          type: 'raster',
          source: 'pm-surface',
          paint: { 'raster-opacity': 0.42, 'raster-resampling': 'linear', 'raster-fade-duration': 0 },
        },
        BEFORE_DATA_LAYER,
      )
      map.addSource('wards', { type: 'geojson', data: wardsGeo })
      map.addSource('ward-labels', { type: 'geojson', data: wardLabels })
      map.addLayer(
        {
          id: 'wards-fill',
          type: 'fill',
          source: 'wards',
          paint: {
            'fill-color': ['get', 'color'],
            'fill-opacity': ['case', ['boolean', ['feature-state', 'hover'], false], 0.28, 0.14],
          },
        },
        BEFORE_DATA_LAYER,
      )
      map.addLayer(
        {
          id: 'wards-line',
          type: 'line',
          source: 'wards',
          paint: {
            'line-color': '#ece7dc',
            'line-opacity': ['interpolate', ['linear'], ['zoom'], 10, 0.14, 13, 0.22],
            'line-width': ['interpolate', ['linear'], ['zoom'], 10, 0.6, 14, 1.4],
            'line-dasharray': [4, 3],
          },
        },
        BEFORE_DATA_LAYER,
      )
      map.addLayer(
        {
          id: 'wards-code',
          type: 'symbol',
          source: 'ward-labels',
          minzoom: 10.6,
          layout: {
            'text-field': ['get', 'code'],
            'text-font': ['Noto Sans Bold'],
            'text-size': 11,
            'text-letter-spacing': 0.1,
            'text-allow-overlap': false,
          },
          paint: { 'text-color': '#ece7dc', 'text-opacity': 0.38, 'text-halo-color': '#0a1119', 'text-halo-width': 1 },
        },
        BEFORE_LABELS,
      )

      let hovered: number | string | undefined
      map.on('mousemove', 'wards-fill', (e) => {
        const f = e.features?.[0]
        if (!f) return
        if (hovered !== undefined) map.setFeatureState({ source: 'wards', id: hovered }, { hover: false })
        hovered = f.id
        if (hovered !== undefined) map.setFeatureState({ source: 'wards', id: hovered }, { hover: true })
      })
      map.on('mouseleave', 'wards-fill', () => {
        if (hovered !== undefined) map.setFeatureState({ source: 'wards', id: hovered }, { hover: false })
        hovered = undefined
      })
      map.on('click', 'wards-fill', (e) => {
        const code = e.features?.[0]?.properties?.code as string | undefined
        if (!code) return
        if (overlayRef.current?.pickObject({ x: e.point.x, y: e.point.y, radius: 4 })) return
        setWard(code)
        setWardPick({ code, x: e.point.x, y: e.point.y })
      })
      setReady(true)
    })

    return () => {
      resizeObserver.disconnect()
      overlayRef.current = null
      mapRef.current = null
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ----- Hour-dependent fields -----
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    renderSurface(data, hour, surfaceCanvas.current)
    const src = map.getSource('pm-surface') as ImageSource | undefined
    src?.updateImage({ url: surfaceCanvas.current.toDataURL(), coordinates: SURFACE_COORDS })
  }, [ready, hour, data])

  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const wards = map.getSource('wards') as GeoJSONSource | undefined
    wards?.setData(wardGeoJSON(data, hour))
  }, [ready, data, hour, layers.wards])

  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    map.setLayoutProperty('pm-surface', 'visibility', layers.surface ? 'visible' : 'none')
    for (const id of ['wards-fill', 'wards-line', 'wards-code']) {
      map.setLayoutProperty(id, 'visibility', layers.wards ? 'visible' : 'none')
    }
    map.setLayoutProperty(BEFORE_DATA_LAYER, 'visibility', layers.buildings ? 'visible' : 'none')
  }, [ready, layers])

  const streamlines = useMemo<Streamline[]>(() => traceStreamlines(data, hour), [data, hour])
  const values = useMemo(() => new Map(stationValues(data, hour).map((v) => [v.id, v.pm])), [data, hour])
  const fluxes = useMemo(() => fluxEdges(data, hour), [data, hour])
  const fluxArcs = useMemo<FluxArc[]>(() => {
    const stations = new Map(data.stations.stations.map((s) => [s.id, s]))
    return fluxes.flatMap((edge) => {
      const src = stations.get(edge.src)
      const dst = stations.get(edge.dst)
      if (!src || !dst) return []
      return [
        {
          ...edge,
          kind: 'airphynet' as const,
          srcName: stationName(src),
          dstName: stationName(dst),
          sourcePosition: [src.lon, src.lat, 80],
          targetPosition: [dst.lon, dst.lat, 80],
        },
      ]
    })
  }, [data, fluxes])

  const staticLayers = useMemo(() => {
    const out: Layer[] = []
    if (layers.sources) {
      out.push(
        new PathLayer({
          id: 'line-sources',
          data: data.wind.line_sources,
          getPath: (d) => d.path,
          getColor: [236, 231, 220, 90],
          getWidth: 2,
          widthUnits: 'pixels',
          getDashArray: [4, 4],
          dashJustified: true,
          extensions: [new PathStyleExtension({ dash: true })],
          pickable: true,
        }),
        new ScatterplotLayer({
          id: 'plume-sources',
          data: data.wind.plume_sources,
          getPosition: (d) => [d.lon, d.lat],
          getRadius: (d) => 350 + d.strength * 500,
          getFillColor: [236, 231, 220, 18],
          getLineColor: [236, 231, 220, 170],
          lineWidthMinPixels: 1.2,
          stroked: true,
          pickable: true,
        }),
      )
    }
    if (layers.facilities) {
      out.push(
        new ScatterplotLayer({
          id: 'facilities',
          data: data.wards.facilities,
          getPosition: (d) => [d.lon, d.lat],
          getRadius: (d) => (d.type === 'hospital' ? 110 : 55),
          radiusMinPixels: 1.5,
          getFillColor: (d) => (d.type === 'hospital' ? [236, 231, 220, 230] : [207, 227, 242, 140]),
          getLineColor: [10, 17, 25, 255],
          lineWidthMinPixels: 0.5,
          stroked: true,
          pickable: true,
        }),
      )
    }
    return out
  }, [data, layers.sources, layers.facilities])

  // ----- deck.gl layers, re-rendered every frame for the wind animation -----
  useEffect(() => {
    const overlay = overlayRef.current
    if (!ready || !overlay) return
    const stations = data.stations.stations.filter((s) => TARGET_SET.has(s.id) || layers.network)
    const targets = stations.filter((s) => TARGET_SET.has(s.id))
    const pm = (id: number) => values.get(id) ?? 0
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const build = (t: number): Layer[] => {
      const pulse = 0.5 + 0.5 * Math.sin(t * 2.2)
      const out: Layer[] = [...staticLayers]
      if (layers.wind) {
        out.push(
          new TripsLayer<Streamline>({
            id: 'wind',
            data: streamlines,
            getPath: (d) => d.path,
            getTimestamps: (d) => d.timestamps,
            getColor: [207, 227, 242],
            opacity: 0.55,
            widthMinPixels: 1.3,
            getWidth: 1.3,
            widthUnits: 'pixels',
            capRounded: true,
            jointRounded: true,
            fadeTrail: true,
            trailLength: 9,
            currentTime: reduce ? WIND_LOOP / 2 : (t * 9) % WIND_LOOP,
          }),
        )
      }
      if (layers.airphynet) {
        out.push(
          new ArcLayer<FluxArc>({
            id: 'airphynet-flux',
            data: fluxArcs,
            getSourcePosition: (d) => d.sourcePosition,
            getTargetPosition: (d) => d.targetPosition,
            getSourceColor: (d) => pmRGB(pm(d.src)),
            getTargetColor: [207, 227, 242, 210],
            getWidth: (d) => 1.2 + Math.min(5, d.flux / 15),
            getHeight: 0.45,
            widthUnits: 'pixels',
            widthMinPixels: 1.2,
            pickable: true,
            updateTriggers: { getSourceColor: values, getWidth: fluxArcs },
          }),
        )
      }
      out.push(
        new ScatterplotLayer({
          id: 'station-halo',
          data: targets,
          getPosition: (d) => [d.lon, d.lat],
          getRadius: (d) => (d.id === stationId ? 1100 + pulse * 250 : 850),
          getFillColor: (d) => [...pmRGB(pm(d.id)), d.id === stationId ? 60 : 38] as [number, number, number, number],
          stroked: false,
          updateTriggers: { getRadius: [stationId, pulse], getFillColor: [stationId, values] },
        }),
        new ScatterplotLayer({
          id: 'station-ring',
          data: targets.filter((s) => s.id === stationId),
          getPosition: (d) => [d.lon, d.lat],
          getRadius: 1100 + pulse * 250,
          filled: false,
          stroked: true,
          getLineColor: [236, 231, 220, 140 - pulse * 80],
          lineWidthMinPixels: 1.2,
          updateTriggers: { getRadius: pulse, getLineColor: pulse },
        }),
        new ColumnLayer({
          id: 'target-columns',
          data: targets,
          diskResolution: 36,
          radius: 380,
          getPosition: (d) => [d.lon, d.lat],
          getElevation: (d) => Math.max(40, pm(d.id) * ELEV_PER_UG),
          getFillColor: (d) => [...pmRGB(pm(d.id)), 240] as [number, number, number, number],
          extruded: true,
          pickable: true,
          material: { ambient: 0.6, diffuse: 0.5, shininess: 32, specularColor: [240, 240, 240] },
          updateTriggers: { getElevation: values, getFillColor: values },
          transitions: reduce ? undefined : { getElevation: 220 },
        }),
        new ColumnLayer({
          id: 'network-columns',
          data: stations.filter((s) => !TARGET_SET.has(s.id)),
          diskResolution: 20,
          radius: 170,
          getPosition: (d) => [d.lon, d.lat],
          getElevation: (d) => Math.max(30, pm(d.id) * ELEV_PER_UG),
          getFillColor: (d) => [...pmRGB(pm(d.id)), 170] as [number, number, number, number],
          extruded: true,
          pickable: true,
          material: { ambient: 0.6, diffuse: 0.5, shininess: 16, specularColor: [200, 200, 200] },
          updateTriggers: { getElevation: values, getFillColor: values },
          transitions: reduce ? undefined : { getElevation: 220 },
        }),
      )
      if (fontsLoaded) {
        out.push(
          new TextLayer({
            id: 'target-values',
            data: targets,
            getPosition: (d) => [d.lon, d.lat, Math.max(40, pm(d.id) * ELEV_PER_UG) + 160],
            getText: (d) => pm(d.id).toFixed(0),
            getSize: (d) => (d.id === stationId ? 30 : 24),
            getColor: (d) => [...pmRGB(pm(d.id)), 255] as [number, number, number, number],
            fontFamily: '"IBM Plex Sans Condensed"',
            fontWeight: 600,
            characterSet: '0123456789.—',
            fontSettings: { sdf: true, fontSize: 64, buffer: 8 },
            outlineWidth: 4,
            outlineColor: [10, 17, 25, 230],
            getTextAnchor: 'middle',
            getAlignmentBaseline: 'bottom',
            billboard: true,
            pickable: true,
            updateTriggers: { getText: values, getPosition: values, getColor: values, getSize: stationId },
          }),
          new TextLayer({
            id: 'target-names',
            data: targets,
            getPosition: (d) => [d.lon, d.lat, Math.max(40, pm(d.id) * ELEV_PER_UG) + 160],
            getText: (d) => stationName(d),
            getSize: 13,
            getColor: (d) => (d.id === stationId ? [236, 231, 220, 255] : [236, 231, 220, 190]),
            fontFamily: '"IBM Plex Sans Condensed"',
            fontWeight: 500,
            characterSet: 'auto',
            fontSettings: { sdf: true, fontSize: 48, buffer: 8 },
            outlineWidth: 4,
            outlineColor: [10, 17, 25, 230],
            getTextAnchor: 'middle',
            getAlignmentBaseline: 'top',
            getPixelOffset: [0, 4],
            billboard: true,
            pickable: true,
            updateTriggers: { getPosition: values, getColor: stationId },
          }),
        )
      }
      return out
    }

    let raf = 0
    const start = performance.now()
    const tick = () => {
      const t = (performance.now() - start) / 1000
      overlay.setProps({
        layers: build(t),
        getCursor: ({ isHovering }) => (isHovering ? 'pointer' : 'grab'),
        onHover: (info: PickingInfo) => setHover(tooltipFor(info)),
        onClick: (info: PickingInfo) => {
          const obj = info.object as { id?: number; role?: string } | undefined
          if (obj && typeof obj.id === 'number' && obj.role) {
            setStation(obj.id)
            setWardPick(null)
            return true
          }
          return false
        },
      })
      if (!reduce) raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [ready, data, streamlines, values, fluxArcs, stationId, staticLayers, layers.wind, layers.airphynet, layers.network, fontsLoaded, setStation])

  function tooltipFor(info: PickingInfo): { x: number; y: number; content: ReactNode } | null {
    const o = info.object as Record<string, unknown> | undefined
    if (!o) return null
    if (o.kind === 'airphynet' && typeof o.flux === 'number' && typeof o.km === 'number') {
      return {
        x: info.x,
        y: info.y,
        content: (
          <>
            <div className="text-[13px] font-medium">
              AirPhyNet flux: {String(o.srcName)} → {String(o.dstName)}
            </div>
            <div className="mt-1 font-cond text-lg tnum">
              {o.flux.toFixed(1)} <span className="font-sans text-2xs text-mute">µg/m²/s</span>
              <span className="ml-2 font-sans text-2xs text-mute">({o.km.toFixed(1)} km)</span>
            </div>
          </>
        ),
      }
    }
    if (typeof o.role === 'string' && typeof o.id === 'number') {
      const v = values.get(o.id)
      const name = stationName(o as { id: number; short: string })
      const role = o.role === 'target' ? 'Forecast station' : `${String(o.operator)} monitor · scaled from forecasts`
      return {
        x: info.x,
        y: info.y,
        content: (
          <>
            <div className="text-[13px] font-medium">{name}</div>
            <div className="text-2xs text-mute">{role}</div>
            <div className="mt-1 font-cond text-lg tnum">
              {v != null ? v.toFixed(1) : '—'} <span className="font-sans text-2xs text-mute">µg/m³</span>
            </div>
          </>
        ),
      }
    }
    if (typeof o.category === 'string') {
      return {
        x: info.x,
        y: info.y,
        content: (
          <>
            <div className="text-[13px] font-medium">{String(o.name)}</div>
            <div className="text-2xs text-mute">{o.category}</div>
          </>
        ),
      }
    }
    if (Array.isArray(o.path)) {
      return {
        x: info.x,
        y: info.y,
        content: (
          <>
            <div className="text-[13px] font-medium">{String(o.name)}</div>
            <div className="text-2xs text-mute">Traffic emission corridor</div>
          </>
        ),
      }
    }
    if (typeof o.type === 'string' && (o.type === 'school' || o.type === 'hospital')) {
      return {
        x: info.x,
        y: info.y,
        content: (
          <>
            <div className="text-[13px] font-medium">
              {o.indicative ? (o.type === 'school' ? 'School cluster' : 'Hospital') : String(o.name)}
            </div>
            <div className="text-2xs text-mute">
              {o.type === 'school' ? 'UDISE+ school' : 'BMC civic hospital'} · ward {String(o.ward)}
            </div>
          </>
        ),
      }
    }
    return null
  }

  // Fly to the selected station when it changes from outside the map.
  const prevStation = useRef(stationId)
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || prevStation.current === stationId) return
    prevStation.current = stationId
    const st = data.stations.stations.find((s) => s.id === stationId)
    if (!st) return
    map.easeTo({ center: [st.lon, st.lat], zoom: Math.max(map.getZoom(), 11.6), duration: 1400, essential: true })
  }, [ready, stationId, data])

  const flyTo = (cam: Cam) => mapRef.current?.flyTo({ ...CAMERAS[cam], duration: 1800, essential: true })

  const wardPm = wardPick ? sampleAt(data.wards.wards.find((w) => w.code === wardPick.code)?.series, hour) : null

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} className="map-root absolute inset-0" aria-label="3D map of Mumbai air quality" />
      {mapError && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="panel max-w-md p-6 text-sm text-mute">
            <div className="mb-1 text-[15px] font-medium text-paper">The 3D map could not start</div>
            {mapError}. WebGL may be disabled in this browser; the Forecast and Ward actions views still work.
          </div>
        </div>
      )}
      <StationCard />
      <LayerPanel onCamera={flyTo} />
      {wardPick && <WardPopover pick={wardPick} pm={wardPm} onClose={() => setWardPick(null)} />}
      {hover && (
        <div
          className="panel pointer-events-none absolute z-20 px-3 py-2"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
        >
          {hover.content}
        </div>
      )}
    </div>
  )
}
