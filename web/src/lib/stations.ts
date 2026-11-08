import type { Block, Station, TwinData } from '../data/types'

/** The four CPCB stations scored on the Zindi leaderboard, north to south. */
export const TARGET_IDS = [11606, 11611, 3409511, 3409510] as const

export const TARGET_META: Record<number, { name: string; area: string; operator: string }> = {
  11606: { name: 'Borivali East', area: 'R/C ward · next to Sanjay Gandhi National Park', operator: 'IITM' },
  11611: { name: 'Malad West', area: 'P/N ward · coastal suburb, creek & high-rise construction', operator: 'IITM' },
  3409511: { name: 'Shivaji Nagar', area: 'M/E ward · 0.8 km downwind of the Deonar landfill', operator: 'BMC' },
  3409510: { name: 'Byculla', area: 'E ward · dense island-city core near the docks', operator: 'BMC' },
}

/** Display name without the operator suffix, e.g. "Malad West (IITM)" → "Malad West". */
export function stationName(s: Pick<Station, 'id' | 'short'>): string {
  return TARGET_META[s.id]?.name ?? s.short.replace(/\s*\([^)]*\)\s*$/, '')
}

export function targetStations(data: TwinData): Station[] {
  return TARGET_IDS.map((id) => data.stations.stations.find((s) => s.id === id)).filter((s): s is Station => !!s)
}

export const BLOCK_LABEL: Record<Block, string> = {
  validation: 'Validation (observed)',
  gap: 'Forecast lead-in',
  public: 'Public leaderboard',
  unscored: 'Forecast',
  private: 'Scored window',
}

/** "Tue 11 Nov" and "08:00" parts of an IST ISO string, without timezone conversion. */
export function istParts(iso: string | undefined): { day: string; date: string; time: string; short: string } {
  if (!iso) return { day: '', date: '—', time: '—', short: '—' }
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number)
  const dow = new Date(Date.UTC(y, m - 1, d)).getUTCDay()
  const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const date = `${d} ${MON[m - 1]}`
  const time = iso.slice(11, 16)
  return { day: DOW[dow], date, time, short: `${date} ${time}` }
}
