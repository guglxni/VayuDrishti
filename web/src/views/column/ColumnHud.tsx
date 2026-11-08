import { ArrowUp } from 'lucide-react'
import type { TwinData } from '../../data/types'
import { pmColor } from '../../lib/aqi'
import { stationPM } from '../../lib/series'
import type { FluxEdge } from '../../lib/physics'
import { stationName, TARGET_IDS, TARGET_META } from '../../lib/stations'
import { useTwin } from '../../state/store'
import { AqiChip, AqiLegend, cx, Eyebrow, PmValue, Stat } from '../../ui/primitives'
import { ColumnChart } from './ColumnChart'
import { fmtInt, TIME_LAPSE, verticalProfileFraction, type ColumnState } from './derive'

function StationPicker({ data, hour, stationId }: { data: TwinData; hour: number; stationId: number }) {
  const setStation = useTwin((s) => s.setStation)
  return (
    <div role="tablist" aria-label="Station" className="grid grid-cols-2 gap-1 rounded-[10px] border border-line bg-ink/60 p-0.5">
      {TARGET_IDS.map((id) => {
        const active = id === stationId
        const pm = stationPM(data, id, hour)
        return (
          <button
            key={id}
            role="tab"
            aria-selected={active}
            onClick={() => setStation(id)}
            className={cx(
              'flex items-center justify-between gap-2 rounded-[8px] px-2.5 py-1.5 text-left text-[12.5px] font-medium transition-colors',
              active ? 'bg-paper text-ink' : 'text-mute hover:bg-paper/5 hover:text-paper',
            )}
          >
            <span className="truncate">{TARGET_META[id].name}</span>
            <span className="flex shrink-0 items-center gap-1 font-cond text-[12px] tnum">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: pmColor(pm) }} />
              {pm != null ? Math.round(pm) : '—'}
            </span>
          </button>
        )
      })}
    </div>
  )
}

function TransportRow({
  label,
  edge,
  peerId,
  nameFor,
}: {
  label: string
  edge: FluxEdge | null
  peerId: number | undefined
  nameFor: (id: number) => string
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-2xs">
      <div className="min-w-0">
        <div className="text-faint">{label}</div>
        <div className={cx('mt-0.5 truncate text-[12px]', edge ? 'text-paper' : 'text-faint')}>
          {peerId != null ? nameFor(peerId) : 'No active transport edge'}
        </div>
      </div>
      {edge && (
        <div className="shrink-0 text-right font-cond tnum">
          <div className="text-[15px] font-medium text-wind">{edge.flux.toFixed(1)}</div>
          <div className="text-[10px] text-faint">{edge.km.toFixed(1)} km</div>
        </div>
      )}
    </div>
  )
}

export function ColumnHud({ data, st, stationId, fluxes }: { data: TwinData; st: ColumnState; stationId: number; fluxes: FluxEdge[] }) {
  const { met } = st
  const ventHint = st.dayVentPeak && met.vent != null ? `Best today ${fmtInt(st.dayVentPeak.value)} at ${st.dayVentPeak.time}` : 'Mixing height × wind speed'
  const incoming = fluxes.find((edge) => edge.dst === stationId) ?? null
  const outgoing = fluxes.find((edge) => edge.src === stationId) ?? null
  const nameFor = (id: number) => {
    const station = data.stations.stations.find((s) => s.id === id)
    return station ? stationName(station) : (TARGET_META[id]?.name ?? String(id))
  }
  const lidM = Math.max(1, met.pblh ?? 400)
  const stratification = 1 + 0.9 * (1 - Math.max(0, Math.min(1, (met.sw ?? 0) / 700)))
  const surfaceFraction = verticalProfileFraction(0, lidM, stratification)
  const lidFraction = verticalProfileFraction(lidM, lidM, stratification)
  const trappingRatio = lidFraction > 0 ? surfaceFraction / lidFraction : null
  return (
    <div
      className="panel absolute left-5 top-20 z-10 flex w-[340px] flex-col overflow-hidden"
      style={{ maxHeight: 'calc(100% - 80px - 180px)' }}
    >
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        <div className="flex items-baseline justify-between">
          <Eyebrow>Air column</Eyebrow>
          <span className="font-mono text-[10px] text-faint tnum">
            {st.parts.day} {st.parts.date} · {st.parts.time} IST
          </span>
        </div>
        <h2 className="mt-1 text-[15px] font-medium leading-snug text-paper">How the air above a station traps PM2.5</h2>

        <div className="mt-3">
          <StationPicker data={data} hour={st.i} stationId={stationId} />
        </div>

        <div className="mt-4">
          <div className="text-[17px] font-medium leading-tight text-paper">{st.stationName}</div>
          <div className="mt-0.5 text-2xs text-mute">{st.area}</div>
        </div>

        <div className="mt-3 flex items-end justify-between gap-3">
          <div>
            <PmValue value={st.pm} size="lg" />
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <AqiChip value={st.pm} />
              {st.pmSource && <span className="text-2xs text-faint">{st.pmSource}</span>}
            </div>
          </div>
          {st.measured != null && st.pmSource !== 'Measured' && (
            <div className="text-right">
              <div className="text-2xs text-faint">Measured</div>
              <div className="font-cond text-lg tnum" style={{ color: pmColor(st.measured) }}>
                {Math.round(st.measured)}
              </div>
            </div>
          )}
        </div>

        <p className="mt-3 text-[13px] leading-relaxed text-paper/85">{st.sentence}</p>

        <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-line pt-3">
          <Stat
            label="Mixing height"
            value={met.pblh != null && met.pblh < 50 ? '< 50' : fmtInt(met.pblh)}
            unit="m"
            hint={st.dayPeak ? `Peak ${fmtInt(st.dayPeak.value)} m at ${st.dayPeak.time}` : undefined}
          />
          <Stat
            label="Wind"
            value={
              <span className="inline-flex items-center gap-1">
                {met.dirFrom != null && (
                  <ArrowUp
                    size={14}
                    className="text-wind"
                    style={{ transform: `rotate(${(met.dirFrom + 180) % 360}deg)` }}
                    aria-hidden
                  />
                )}
                {met.speed != null ? met.speed.toFixed(1) : '—'}
              </span>
            }
            unit="m/s"
            hint={st.dirText ? `From ${st.dirText}${st.exposure ? ` · ${st.exposure}` : ''}` : undefined}
          />
          <Stat label="Ventilation" value={fmtInt(met.vent)} unit="m²/s" hint={ventHint} />
          <Stat
            label="Sunlight"
            value={met.sw != null ? fmtInt(met.sw) : '—'}
            unit="W/m²"
            hint={st.sunEl > 0 ? `Sun ${Math.round(st.sunEl)}° up` : 'Sun below horizon'}
          />
        </div>
        <div className="mt-3 flex items-baseline gap-2 text-2xs">
          <span className="text-faint">Air pattern</span>
          <span className="text-paper/85">{st.regime}</span>
        </div>

        <div className="mt-4 border-t border-line pt-3">
          <div className="flex items-baseline justify-between gap-2">
            <Eyebrow>AirPhyNet transport</Eyebrow>
            <span className="font-mono text-[10px] text-faint">µg/m²/s</span>
          </div>
          <div className="mt-2 space-y-2">
            <TransportRow label="Strongest incoming" edge={incoming} peerId={incoming?.src} nameFor={nameFor} />
            <TransportRow label="Strongest outgoing" edge={outgoing} peerId={outgoing?.dst} nameFor={nameFor} />
          </div>
          <div className="mt-3 flex items-end justify-between gap-3 border-t border-line pt-2.5">
            <div>
              <div className="text-2xs text-faint">Illustrative profile · fixed C_surface / C_lid</div>
              <div className="mt-0.5 text-[10px] leading-relaxed text-mute">Model shape uses a fixed 35% drawdown at the lid.</div>
              <div className="mt-0.5 text-[10px] leading-relaxed text-mute">
                {st.pm != null
                  ? `Surface ${(st.pm * surfaceFraction).toFixed(1)} · lid ${(st.pm * lidFraction).toFixed(1)} µg/m³`
                  : 'Normalised vertical profile · PM2.5 unavailable'}
              </div>
            </div>
            <div className="shrink-0 font-cond text-[24px] font-semibold leading-none tnum text-wind">
              {trappingRatio != null ? `${trappingRatio.toFixed(2)}×` : '—'}
            </div>
          </div>
        </div>

        <div className="mt-4 border-t border-line pt-3">
          <ColumnChart data={data} stationId={stationId} hour={st.i} />
        </div>

        <div className="mt-4 space-y-2 border-t border-line pt-3">
          <AqiLegend compact />
          <p className="text-2xs leading-relaxed text-faint">
            Particle colour is PM2.5 on the CPCB scale, and more particles mean more PM2.5. Pale streaks show the wind. Drawn to scale (1 km
            ring, heights in metres, no vertical exaggeration); motion is sped up about {TIME_LAPSE}×. Buildings and parks are stylised
            from Open Buildings density and AlphaEarth roughness and green-cover indices.
            {met.usedCity ? ' Mixing height uses the city average for this hour.' : ''}
          </p>
        </div>
      </div>
    </div>
  )
}
