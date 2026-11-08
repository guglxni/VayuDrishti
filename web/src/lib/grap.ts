import type { Facility, TwinData, Ward } from '../data/types'
import { grapStage } from './aqi'
import { cityMet, REGIME_LABEL } from './physics'
import { fmtHour, sampleAt } from './series'

export interface WardContext {
  ward: Ward
  pm: number
  peak24: number
  hourIso: string
  met: ReturnType<typeof cityMet>
  facilities: Facility[]
}

export function wardContext(data: TwinData, ward: Ward, f: number): WardContext {
  const pm = sampleAt(ward.series, f) ?? ward.private_mean
  const i = Math.round(f)
  const next = ward.series.slice(i, i + 24).filter((v): v is number => v != null)
  return {
    ward,
    pm,
    peak24: next.length ? Math.max(...next) : pm,
    hourIso: data.forecasts.hours[i],
    met: cityMet(data, f),
    facilities: data.wards.facilities.filter((x) => x.ward === ward.code && !x.indicative),
  }
}

/** Offline, physics-grounded GRAP directive that recomputes for the scrubbed hour. */
export function offlineDirective(ctx: WardContext): string {
  const { ward, pm, peak24, met } = ctx
  const g = grapStage(Math.max(pm, peak24 * 0.85))
  const vcState = met.vc < 1200 ? 'suppressed — below the 1,200 m²/s stagnation threshold' : met.vc < 3000 ? 'marginal' : 'active'
  const lines: string[] = []
  lines.push(`### Ward ${ward.code} · ${ward.name}`)
  lines.push(
    `**${fmtHour(ctx.hourIso)}** · Trigger **${g.stage}** · ward PM2.5 **${pm.toFixed(1)} µg/m³** now, **${peak24.toFixed(1)}** peak in next 24 h ` +
      `(${(pm / 60).toFixed(2)}× NAAQS · ${(pm / 15).toFixed(1)}× WHO)`,
  )
  lines.push('')
  lines.push('#### Atmosphere')
  lines.push(
    `- Boundary layer **${met.pblh.toFixed(0)} m**, wind **${met.wind.toFixed(1)} m/s** from **${met.dirFrom.toFixed(0)}°** → ventilation **${met.vc.toFixed(0)} m²/s** (${vcState}).`,
  )
  lines.push(`- Regime: *${REGIME_LABEL[met.regime] ?? met.regime}*. Onshore component ${met.onshore.toFixed(1)} m/s; Deonar-axis transport ${met.deonar.toFixed(1)} m/s.`)
  lines.push('')
  lines.push('#### Actions')
  if (g.level >= 1) {
    lines.push(`- Issue dust-control notices to all **${ward.maharera_sites} MahaRERA sites**: mist guns + wheel-wash 05:30–09:30 IST; halt demolition if hourly PM2.5 > 90.`)
  } else {
    lines.push(`- Keep **${ward.maharera_sites} MahaRERA sites** on routine sensor-compliance watch; no curtailment needed.`)
  }
  if (['M/E', 'M/W', 'N', 'L'].includes(ward.code) && (met.regime === 'land_breeze' || met.deonar > 1)) {
    lines.push('- Deonar landfill: thermal-drone smoulder patrol and perimeter fogging on GMLR before dawn — land breeze is carrying smoke toward Shivaji Nagar.')
  }
  if (['E', 'F/S', 'B', 'G/S'].includes(ward.code)) {
    lines.push('- Port/freight corridor: restrict multi-axle diesel staging on P. D\'Mello Rd & Mazgaon dock approaches 07:00–11:00 and 19:00–23:00 IST.')
  }
  if (['P/N', 'K/E', 'K/W', 'R/C', 'R/S', 'R/N', 'P/S'].includes(ward.code)) {
    lines.push('- Western Express Highway / Link Road: green-wave signal timing until the afternoon sea-breeze front arrives.')
  }
  if (met.pblh < 300) {
    lines.push('- Night/early-morning inversion: move mechanical road sweeping to 03:30–06:30 IST with water suppression; avoid dry sweeping.')
  }
  lines.push('')
  lines.push('#### People')
  lines.push(
    `- **${ward.pop_2026.toLocaleString('en-IN')}** residents, **${ward.vulnerable_pop.toLocaleString('en-IN')}** children <5 / elderly >60 (WorldPop 2026, doi:10.71646/AAGK0G).`,
  )
  if (g.level >= 1) {
    lines.push(`- **${ward.udise_schools} UDISE+ schools**: indoor assembly and no outdoor PE until 13:30 IST. **${ward.civic_hospitals} civic hospitals**: pre-stage bronchodilator and nebuliser stock.`)
  } else {
    lines.push(`- **${ward.udise_schools} UDISE+ schools** and **${ward.civic_hospitals} civic hospitals** stay on routine hourly telemetry watch.`)
  }
  for (const f of ctx.facilities.slice(0, 3)) {
    lines.push(`- ${f.name}${f.protocol ? ` — ${f.protocol}` : ''}`)
  }
  return lines.join('\n')
}

export function geminiPrompt(ctx: WardContext, question: string): { system: string; prompt: string } {
  const { ward, pm, peak24, met } = ctx
  const system =
    'You are the BMC (Brihanmumbai Municipal Corporation) Graded Response Action Plan copilot. ' +
    'Write concise, operational, ward-specific directives in Markdown. Ground every claim in the numbers provided; ' +
    'do not invent measurements. Use IST times. Cite ClimateVerse DOIs given in the context when relevant. Max 250 words.'
  const prompt = [
    `Ward ${ward.code} (${ward.name}), ${ward.zone}.`,
    `Time: ${ctx.hourIso}. Forecast ward PM2.5 now ${pm.toFixed(1)} µg/m³, next-24h peak ${peak24.toFixed(1)} µg/m³ (India NAAQS 24h = 60, WHO = 15).`,
    `Boundary layer ${met.pblh.toFixed(0)} m, wind ${met.wind.toFixed(1)} m/s from ${met.dirFrom.toFixed(0)}°, ventilation coefficient ${met.vc.toFixed(0)} m²/s, regime ${met.regime}.`,
    `Population ${ward.pop_2026} (vulnerable ${ward.vulnerable_pop}), slum share ${ward.slum_pct}%, MahaRERA construction sites ${ward.maharera_sites}, UDISE+ schools ${ward.udise_schools}, civic hospitals ${ward.civic_hospitals}.`,
    `AlphaEarth impervious index ${ward.alphaearth_impervious}, Open Buildings density ${ward.open_bldg_density}.`,
    `Named facilities: ${ctx.facilities.map((f) => f.name).join('; ') || 'none listed'}.`,
    'DOIs: WorldPop doi:10.71646/AAGK0G, UDISE+ doi:10.71646/XJIY8W, BMC hospitals doi:10.71646/1XSUVR, MahaRERA doi:10.71646/4ULQRE, Deonar landfill doi:10.71646/MWQNR8, Open-Meteo doi:10.71646/XXOBKO.',
    '',
    question.trim() ? `Officer's question: ${question.trim()}` : 'Produce the GRAP directive for the next 24 hours.',
  ].join('\n')
  return { system, prompt }
}
