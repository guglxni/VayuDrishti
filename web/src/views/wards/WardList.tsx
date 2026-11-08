import { Search, X } from 'lucide-react'
import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import type { Ward } from '../../data/types'
import { grapStage, pmColor } from '../../lib/aqi'
import { fmt, sampleAt } from '../../lib/series'
import { useData, useTwin } from '../../state/store'
import { cx, Eyebrow } from '../../ui/primitives'
import { GrapChip, Sparkline } from './parts'
import { whenLabel } from './util'

interface Row {
  ward: Ward
  pm: number | null
}

const STAGES = [
  { level: 3, label: 'Stage III', color: grapStage(120).color },
  { level: 2, label: 'Stage II', color: grapStage(90).color },
  { level: 1, label: 'Stage I', color: grapStage(60).color },
  { level: 0, label: 'Watch', color: grapStage(0).color },
] as const

export function WardList() {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const wardCode = useTwin((s) => s.wardCode)
  const setWard = useTwin((s) => s.setWard)
  const [query, setQuery] = useState('')
  const buttons = useRef(new Map<string, HTMLButtonElement>())

  const ranked = useMemo<Row[]>(() => {
    const rows = data.wards.wards.map((ward) => ({ ward, pm: sampleAt(ward.series, hour) }))
    return rows.sort((a, b) => (b.pm ?? -1) - (a.pm ?? -1))
  }, [data, hour])

  const rankOf = useMemo(() => new Map(ranked.map((r, i) => [r.ward.code, i + 1])), [ranked])

  const q = query.trim().toLowerCase()
  const rows = q
    ? ranked.filter((r) => `${r.ward.code} ${r.ward.name} ${r.ward.zone}`.toLowerCase().includes(q))
    : ranked

  const counts = useMemo(() => {
    const c = new Map<number, number>()
    for (const r of ranked) {
      const lvl = grapStage(r.pm).level
      c.set(lvl, (c.get(lvl) ?? 0) + 1)
    }
    return c
  }, [ranked])
  const noData = counts.get(-1) ?? 0

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp' && e.key !== 'Home' && e.key !== 'End') return
    if (!rows.length) return
    e.preventDefault()
    const i = rows.findIndex((r) => r.ward.code === wardCode)
    let next = i
    if (e.key === 'ArrowDown') next = i < 0 ? 0 : Math.min(rows.length - 1, i + 1)
    if (e.key === 'ArrowUp') next = i < 0 ? 0 : Math.max(0, i - 1)
    if (e.key === 'Home') next = 0
    if (e.key === 'End') next = rows.length - 1
    const code = rows[next].ward.code
    setWard(code)
    buttons.current.get(code)?.focus()
  }

  return (
    <div className="flex min-h-full flex-col">
      <div className="sticky top-0 z-10 bg-ink px-4 pb-3 pt-1">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-[15px] font-semibold tracking-tight text-paper">Wards by PM2.5 now</h2>
          <span className="font-mono text-2xs tnum text-faint">{whenLabel(data.forecasts.hours[hour])}</span>
        </div>

        <div className="mt-3" aria-label="Wards by GRAP stage this hour">
          <div className="flex h-1.5 gap-px overflow-hidden rounded-full bg-paper/5">
            {STAGES.slice()
              .reverse()
              .map((s) => {
                const n = counts.get(s.level) ?? 0
                if (!n) return null
                return <div key={s.level} style={{ flex: n, background: s.color, opacity: 0.85 }} />
              })}
            {noData > 0 && <div style={{ flex: noData }} className="bg-paper/10" />}
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-2xs text-mute">
            {STAGES.map((s) => (
              <span key={s.level} className={cx('tnum', !(counts.get(s.level) ?? 0) && 'text-faint')}>
                {s.label} <span className="font-cond font-medium text-paper">{counts.get(s.level) ?? 0}</span>
              </span>
            ))}
            {noData > 0 && (
              <span className="tnum">
                No data <span className="font-cond font-medium text-paper">{noData}</span>
              </span>
            )}
          </div>
        </div>

        <label className="relative mt-3 block">
          <span className="sr-only">Search wards</span>
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setQuery('')
              if (e.key === 'ArrowDown' && rows.length) {
                e.preventDefault()
                const code = rows.some((r) => r.ward.code === wardCode) ? wardCode : rows[0].ward.code
                setWard(code)
                buttons.current.get(code)?.focus()
              }
            }}
            placeholder="Search by ward, area or zone"
            className="h-8 w-full rounded-[10px] border border-line bg-ink-2 pl-8 pr-8 text-[13px] text-paper placeholder:text-faint focus:border-line-strong focus:outline-none"
          />
          {query && (
            <button
              aria-label="Clear search"
              onClick={() => setQuery('')}
              className="absolute right-1.5 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded text-faint hover:text-paper"
            >
              <X size={13} />
            </button>
          )}
        </label>
        <div className="mt-3 flex items-center justify-between text-[10px] text-faint">
          <Eyebrow className="!text-[10px]">Ward</Eyebrow>
          <div className="flex items-center gap-3">
            <Eyebrow className="!text-[10px]">Next 72 h</Eyebrow>
            <Eyebrow className="w-[62px] text-right !text-[10px]">µg/m³ · stage</Eyebrow>
          </div>
        </div>
      </div>

      <div role="listbox" aria-label="Wards" onKeyDown={onKeyDown} className="flex-1 px-2 pb-4">
        {rows.map(({ ward, pm }) => {
          const active = ward.code === wardCode
          return (
            <button
              key={ward.code}
              ref={(el) => {
                if (el) buttons.current.set(ward.code, el)
                else buttons.current.delete(ward.code)
              }}
              role="option"
              aria-selected={active}
              tabIndex={active || (!rows.some((r) => r.ward.code === wardCode) && rows[0].ward.code === ward.code) ? 0 : -1}
              onClick={() => setWard(ward.code)}
              className={cx(
                'group relative flex w-full items-center gap-3 rounded-[10px] py-2 pl-3 pr-2 text-left transition-colors',
                active ? 'bg-paper/8' : 'hover:bg-paper/4',
              )}
            >
              <span
                aria-hidden
                className={cx('absolute left-0 top-2 bottom-2 w-[2px] rounded-full', active ? 'bg-paper' : 'bg-transparent')}
              />
              <span className="w-5 shrink-0 text-right font-cond text-2xs tnum text-faint">{rankOf.get(ward.code)}</span>
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline gap-1.5">
                  <span className={cx('font-mono text-2xs', active ? 'text-paper' : 'text-mute')}>{ward.code}</span>
                  <span className={cx('truncate text-[13px]', active ? 'text-paper' : 'text-paper/85')}>{ward.name}</span>
                </span>
                <span className="block truncate text-2xs text-faint">{ward.zone}</span>
              </span>
              <Sparkline series={ward.series} from={hour} width={56} height={20} />
              <span className="flex w-[62px] shrink-0 flex-col items-end gap-0.5">
                <span className="font-cond text-[17px] font-semibold leading-none tnum" style={{ color: pmColor(pm) }}>
                  {fmt(pm, 0)}
                </span>
                <GrapChip value={pm} className="!px-1.5 !py-0 !text-[10px]" />
              </span>
            </button>
          )
        })}
        {rows.length === 0 && (
          <div className="px-3 py-8 text-center text-[13px] text-mute">
            No ward matches “{query}”.{' '}
            <button className="text-paper underline underline-offset-2" onClick={() => setQuery('')}>
              Clear search
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
