/**
 * CPCB National Air Quality Index (NAQI) sub-index bands for PM2.5 (µg/m³, 24-h basis),
 * using the CPCB colour sequence (dark green → light green → yellow → orange → red → maroon),
 * tuned for legibility on a dark map.
 */
export const AQI_BANDS = [
  { min: 0, max: 30, label: 'Good', color: '#3DAE6B' },
  { min: 30, max: 60, label: 'Satisfactory', color: '#A7CF4E' },
  { min: 60, max: 90, label: 'Moderate', color: '#EFD143' },
  { min: 90, max: 120, label: 'Poor', color: '#F0973A' },
  { min: 120, max: 250, label: 'Very poor', color: '#E2483F' },
  { min: 250, max: Infinity, label: 'Severe', color: '#B02A41' },
] as const

export const NAAQS_24H = 60
export const WHO_24H = 15

type RGB = [number, number, number]

const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]

// Continuous ramp anchored at band mid-points so colours change smoothly on surfaces.
const STOPS: [number, RGB][] = [
  [0, hex('#2F9E62')],
  [22, hex('#3DAE6B')],
  [45, hex('#A7CF4E')],
  [75, hex('#EFD143')],
  [105, hex('#F0973A')],
  [170, hex('#E2483F')],
  [280, hex('#B02A41')],
]

export function pmRGB(pm: number | null | undefined): RGB {
  if (pm == null || !Number.isFinite(pm)) return [95, 107, 120]
  if (pm <= STOPS[0][0]) return STOPS[0][1]
  for (let i = 1; i < STOPS.length; i++) {
    const [x1, c1] = STOPS[i]
    const [x0, c0] = STOPS[i - 1]
    if (pm <= x1) {
      const t = (pm - x0) / (x1 - x0)
      return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t]
    }
  }
  return STOPS[STOPS.length - 1][1]
}

export function pmColor(pm: number | null | undefined, alpha = 1): string {
  const [r, g, b] = pmRGB(pm)
  return alpha >= 1 ? `rgb(${r | 0},${g | 0},${b | 0})` : `rgba(${r | 0},${g | 0},${b | 0},${alpha})`
}

export function pmBand(pm: number | null | undefined): { label: string; color: string } {
  if (pm == null || !Number.isFinite(pm)) return { label: 'No data', color: '#5F6B78' }
  return AQI_BANDS.find((b) => pm < b.max) ?? AQI_BANDS[AQI_BANDS.length - 1]
}

/** BMC Graded Response Action Plan stage, using the thresholds exported with the ward data. */
export function grapStage(pm: number | null | undefined): { stage: string; short: string; color: string; level: number } {
  if (pm == null || !Number.isFinite(pm)) return { stage: 'Unknown', short: '—', color: '#5F6B78', level: -1 }
  if (pm >= 120) return { stage: 'Stage III', short: 'III', color: '#E2483F', level: 3 }
  if (pm >= 90) return { stage: 'Stage II', short: 'II', color: '#F0973A', level: 2 }
  if (pm >= 60) return { stage: 'Stage I', short: 'I', color: '#EFD143', level: 1 }
  return { stage: 'Watch', short: 'W', color: '#A7CF4E', level: 0 }
}
