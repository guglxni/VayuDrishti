import { useMemo, useState } from 'react'
import type { Ward } from '../../data/types'
import { pmColor } from '../../lib/aqi'
import { fmt, sampleAt } from '../../lib/series'

const W = 220

/** Equirectangular locator of the 24 approximate ward cells, scaled by cos(latitude). */
export function Locator({
  wards,
  outline,
  hour,
  selected,
  onSelect,
}: {
  wards: Ward[]
  outline: [number, number][]
  hour: number
  selected: string
  onSelect: (code: string) => void
}) {
  const [hover, setHover] = useState<string | null>(null)

  const proj = useMemo(() => {
    const pts = [...outline, ...wards.flatMap((w) => w.polygon)]
    const lons = pts.map((p) => p[0])
    const lats = pts.map((p) => p[1])
    const minLon = Math.min(...lons)
    const maxLon = Math.max(...lons)
    const minLat = Math.min(...lats)
    const maxLat = Math.max(...lats)
    const kx = Math.cos((((minLat + maxLat) / 2) * Math.PI) / 180)
    const pad = 6
    const s = (W - pad * 2) / ((maxLon - minLon) * kx)
    const H = (maxLat - minLat) * s + pad * 2
    const p = ([lon, lat]: [number, number]) => [pad + (lon - minLon) * kx * s, pad + (maxLat - lat) * s] as const
    const d = (ring: [number, number][]) =>
      ring.map((pt, i) => `${i ? 'L' : 'M'}${p(pt)[0].toFixed(1)},${p(pt)[1].toFixed(1)}`).join('') + 'Z'
    return { H, p, d, outline: d(outline), wards: wards.map((w) => ({ code: w.code, d: d(w.polygon), c: p([w.lon, w.lat]) })) }
  }, [wards, outline])

  const pm = useMemo(() => new Map(wards.map((w) => [w.code, sampleAt(w.series, hour)])), [wards, hour])
  const byCode = useMemo(() => new Map(wards.map((w) => [w.code, w])), [wards])
  const sel = proj.wards.find((w) => w.code === selected)
  const hov = hover && hover !== selected ? proj.wards.find((w) => w.code === hover) : null
  const focus = hov ?? sel
  const focusWard = focus ? byCode.get(focus.code) : undefined

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${proj.H.toFixed(1)}`} className="block h-auto w-full" role="group" aria-label="Ward locator map">
        <path d={proj.outline} fill="#ECE7DC" fillOpacity={0.02} stroke="#ECE7DC" strokeOpacity={0.22} strokeWidth={0.8} />
        {proj.wards.map((w) => {
          const v = pm.get(w.code) ?? null
          const isSel = w.code === selected
          if (isSel) return null
          return (
            <path
              key={w.code}
              d={w.d}
              fill={pmColor(v, hover === w.code ? 0.28 : 0.12)}
              stroke="#ECE7DC"
              strokeOpacity={hover === w.code ? 0.45 : 0.14}
              strokeWidth={0.6}
              className="cursor-pointer transition-[fill-opacity]"
              onPointerEnter={() => setHover(w.code)}
              onPointerLeave={() => setHover((h) => (h === w.code ? null : h))}
              onClick={() => onSelect(w.code)}
            >
              <title>{`${w.code} · ${byCode.get(w.code)?.name} · ${fmt(v, 0)} µg/m³`}</title>
            </path>
          )
        })}
        {sel && (
          <g>
            <path d={sel.d} fill={pmColor(pm.get(sel.code), 0.38)} stroke="#ECE7DC" strokeWidth={1.4} strokeLinejoin="round" />
            <circle cx={sel.c[0]} cy={sel.c[1]} r={1.8} fill="#ECE7DC" />
          </g>
        )}
      </svg>
      {focusWard && (
        <div className="pointer-events-none absolute bottom-0 right-0 max-w-[55%] text-right text-2xs leading-tight">
          <div className="font-mono text-paper">{focusWard.code}</div>
          <div className="text-faint">{hov ? `${fmt(pm.get(focusWard.code), 0)} µg/m³` : focusWard.zone}</div>
        </div>
      )}
    </div>
  )
}
