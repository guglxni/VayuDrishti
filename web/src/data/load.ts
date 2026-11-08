import type { TwinData } from './types'

const FILES = {
  forecasts: 'forecasts_72h.json',
  stations: 'stations.json',
  wind: 'wind_and_plumes.json',
  wards: 'wards_and_civic.json',
  catalog: 'climateverse_catalog.json',
} as const

export async function loadTwinData(onProgress: (p: number) => void): Promise<TwinData> {
  const base = `${import.meta.env.BASE_URL}data/`
  let done = 0
  const keys = Object.keys(FILES) as (keyof typeof FILES)[]
  const entries = await Promise.all(
    keys.map(async (k) => {
      const r = await fetch(base + FILES[k])
      if (!r.ok) throw new Error(`${FILES[k]} → HTTP ${r.status}. Run: .venv/bin/python web/export_web_data.py`)
      const json = await r.json()
      done += 1
      onProgress(done / keys.length)
      return [k, json] as const
    }),
  )
  return Object.fromEntries(entries) as unknown as TwinData
}
