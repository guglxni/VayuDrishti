import { create } from 'zustand'
import type { TwinData } from '../data/types'
import { loadTwinData } from '../data/load'

export type View = 'map' | 'column' | 'forecast' | 'wards'

export type LayerKey = 'surface' | 'wind' | 'wards' | 'network' | 'buildings' | 'sources' | 'facilities'

export const LAYER_LABELS: Record<LayerKey, { label: string; hint: string }> = {
  surface: { label: 'PM2.5 surface', hint: 'Interpolated from 18 stations' },
  wind: { label: 'Wind', hint: 'Open-Meteo 10 m + VABB METAR' },
  wards: { label: 'Ward boundaries', hint: '24 BMC wards (approximate)' },
  network: { label: 'Other stations', hint: '14 CPCB / MPCB / IITM monitors' },
  buildings: { label: '3D buildings', hint: 'OpenStreetMap footprints' },
  sources: { label: 'Emission sources', hint: 'Landfills, refineries, highways' },
  facilities: { label: 'Schools & hospitals', hint: 'UDISE+ and BMC civic hospitals' },
}

const VIEWS: View[] = ['map', 'column', 'forecast', 'wards']

function initialView(): View {
  const v = new URLSearchParams(window.location.search).get('view') as View | null
  return v && VIEWS.includes(v) ? v : 'map'
}

interface TwinState {
  data: TwinData | null
  progress: number
  error: string | null
  load: () => Promise<void>

  view: View
  setView: (v: View) => void

  /** Index into data.forecasts.hours (hour_start, IST). */
  hour: number
  setHour: (h: number | ((prev: number) => number)) => void
  playing: boolean
  setPlaying: (p: boolean | ((prev: boolean) => boolean)) => void

  stationId: number
  setStation: (id: number) => void
  wardCode: string
  setWard: (code: string) => void

  layers: Record<LayerKey, boolean>
  toggleLayer: (k: LayerKey) => void

  aboutOpen: boolean
  setAboutOpen: (o: boolean) => void
}

export const useTwin = create<TwinState>((set, get) => ({
  data: null,
  progress: 0,
  error: null,
  async load() {
    if (get().data) return
    try {
      const data = await loadTwinData((p) => set({ progress: p }))
      const priv = data.forecasts.windows.find((w) => w.id === 'private')
      // Open on the scored window's morning rush hour (Nov 11, 08:00 IST).
      const hour = priv ? priv.start + 8 : 0
      set({ data, hour })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  view: initialView(),
  setView: (view) => {
    const url = new URL(window.location.href)
    if (view === 'map') url.searchParams.delete('view')
    else url.searchParams.set('view', view)
    window.history.replaceState(null, '', url)
    set({ view })
  },

  hour: 0,
  setHour: (h) =>
    set((s) => {
      const max = (s.data?.forecasts.hours.length ?? 1) - 1
      const next = typeof h === 'function' ? h(s.hour) : h
      return { hour: Math.max(0, Math.min(max, Math.round(next))) }
    }),
  playing: false,
  setPlaying: (p) => set((s) => ({ playing: typeof p === 'function' ? p(s.playing) : p })),

  stationId: 3409511,
  setStation: (stationId) => {
    const st = get().data?.stations.stations.find((s) => s.id === stationId)
    set(st ? { stationId, wardCode: st.ward } : { stationId })
  },
  wardCode: 'M/E',
  setWard: (wardCode) => set({ wardCode }),

  layers: {
    surface: true,
    wind: true,
    wards: true,
    network: true,
    buildings: true,
    sources: false,
    facilities: false,
  },
  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),

  aboutOpen: false,
  setAboutOpen: (aboutOpen) => set({ aboutOpen }),
}))

/** Non-null data accessor for components rendered after load. */
export function useData(): TwinData {
  const d = useTwin((s) => s.data)
  if (!d) throw new Error('Twin data not loaded')
  return d
}
