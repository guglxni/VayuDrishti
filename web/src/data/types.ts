export type Num = number | null

export type Block = 'validation' | 'public' | 'unscored' | 'private' | 'gap'

export interface Window {
  id: Block
  start: number
  end: number
  n: number
}

export interface StationForecast {
  obs: Num[]
  obs_clean: Num[]
  models: Record<string, Num[]>
  deciles: (Num[] | null)[]
  q21: (Num[] | null)[]
  median: Num[]
  level: Num[]
  diurnal: Num[]
  cams: Num[]
}

export interface HampelEvent {
  hour_start: string
  raw: number
  clean: number
  stage: 1 | 2
  flag_high_1000: boolean
  context: { hours: string[]; raw: Num[]; clean: Num[] }
}

export interface Analog {
  date: string
  similarity: number
  distance: number
  pm25_mean: number
  curve: Num[]
  wind_ms: number
  pblh_m: number
  rh: number
}

export interface AnalogDay {
  target_met: { wind_ms: number; pblh_m: number; rh: number; temp_c: number }
  analogs: Analog[]
  anen_curve: Num[]
}

export interface LeaderRow {
  key: string
  name: string
  rmse24: number
  rmse72: number | null
  folds24: number[]
}

export interface Forecasts {
  meta: {
    generated_at: string
    cutoff_t0: string
    deciles: number[]
    quantiles21: number[]
    model_names: Record<string, string>
    ensemble_weights: Record<string, { weights: Record<string, number>; shrinkage_alpha: number; oof_rmse_after_shrinkage: number }>
    leaderboard: LeaderRow[]
    provenance: Record<string, string>
  }
  hours: string[]
  blocks: Block[]
  horizon: number[]
  windows: Window[]
  stations: Record<string, StationForecast>
  hampel: Record<string, { n_rows: number; n_stage1: number; n_stage2: number; max_raw: number; events: HampelEvent[] }>
  analogs: Record<string, Record<string, AnalogDay>>
}

export interface Station {
  id: number
  name: string
  short: string
  role: 'target' | 'auxiliary'
  operator: string
  lat: number
  lon: number
  ward: string
  regime?: string
  morphology: Record<string, number | null>
  stats: Record<string, number | string | null>
  diurnal: Num[] | null
  series?: Num[]
  obs?: Num[]
}

export interface StationsFile {
  meta: Record<string, unknown>
  stations: Station[]
}

export interface WindPoint {
  lat: number
  lon: number
  u: Num[]
  v: Num[]
  pblh?: Num[]
  temp?: Num[]
  rh?: Num[]
  sw?: Num[]
  observed?: boolean
}

export type Regime = 'sea_breeze' | 'land_breeze' | 'transitional' | 'stagnant' | 'unknown'

export interface PlumeSource {
  name: string
  category: string
  doi: string
  lat: number
  lon: number
  ward: string
  strength: number
  description: string
}

export interface WindFile {
  meta: Record<string, unknown>
  points: Record<string, WindPoint>
  city: {
    pblh: Num[]
    wind_ms: Num[]
    dir_from: Num[]
    u: Num[]
    v: Num[]
    vc: Num[]
    onshore: Num[]
    deonar_transport: Num[]
    sw: Num[]
    regime: Regime[]
  }
  airphynet: {
    edges: { src: number; dst: number; km: number; bearing: number; mean_alignment: number }[]
    formula: string
    kappa: number
  }
  plume_sources: PlumeSource[]
  line_sources: { name: string; doi: string; strength: number; path: [number, number][] }[]
}

export interface Ward {
  code: string
  name: string
  zone: string
  lat: number
  lon: number
  polygon: [number, number][]
  pop_2026: number
  vulnerable_pop: number
  children_pct: number
  elderly_pct: number
  slum_pct: number
  maharera_sites: number
  alphaearth_impervious: number
  open_bldg_density: number
  udise_schools: number
  civic_hospitals: number
  series: Num[]
  private_mean: number
  private_peak: number
  private_peak_at: string
  naaqs_x: number
  who_x: number
  naaqs_exceed_days: number
  who_exceed_days: number
  forecast_days: number
  grap_stage: string
  directive_md: string
}

export interface Facility {
  id: string
  name: string
  type: 'school' | 'hospital'
  doi: string
  ward: string
  lat: number
  lon: number
  tier?: string
  protocol?: string
  daily_population?: number
  indicative: boolean
}

export interface WardsFile {
  meta: Record<string, unknown>
  wards: Ward[]
  facilities: Facility[]
  outline: [number, number][]
}

export interface Dataset {
  doi: string
  name: string
  url: string
  published_at: string | null
  publisher: string
  subjects: string[]
  file_count: number | null
  category: string
  curated_category: string | null
  description: string
  integrated: boolean
  role: string | null
}

export interface CatalogFile {
  meta: { count: number; integrated: number; source: string }
  datasets: Dataset[]
}

export interface TwinData {
  forecasts: Forecasts
  stations: StationsFile
  wind: WindFile
  wards: WardsFile
  catalog: CatalogFile
}
