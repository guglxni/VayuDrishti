import { useMemo, useState } from 'react'
import { ExternalLink, Search } from 'lucide-react'
import { useData } from '../state/store'
import { cx, Segmented } from './primitives'

/** Searchable ClimateVerse catalogue, shared by the About drawer and Ward actions view. */
export function CatalogExplorer() {
  const data = useData()
  const [query, setQuery] = useState('')
  const [onlyIntegrated, setOnlyIntegrated] = useState(true)
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return data.catalog.datasets
      .filter(
        (dataset) =>
          (!onlyIntegrated || dataset.integrated) &&
          (!needle ||
            `${dataset.name} ${dataset.description} ${dataset.doi} ${dataset.category} ${dataset.role ?? ''} ${dataset.subjects.join(' ')}`
              .toLowerCase()
              .includes(needle)),
      )
      .sort((a, b) => Number(b.integrated) - Number(a.integrated))
  }, [data.catalog.datasets, query, onlyIntegrated])

  return (
    <div aria-label="ClimateVerse data catalog">
      <p className="text-[13px] leading-relaxed text-mute">
        <span className="font-cond font-semibold tnum text-paper">{data.catalog.meta.integrated}</span> of{' '}
        <span className="font-cond tnum text-paper">{data.catalog.meta.count}</span> ClimateVerse India datasets feed the forecasts and ward
        tools. Browse the integrated DOIs, or switch to the full regional catalog.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2 sm:flex-nowrap">
        <label className="flex min-w-[180px] flex-1 items-center gap-2 rounded-[10px] border border-line bg-ink px-3 py-1.5">
          <Search size={14} className="shrink-0 text-faint" aria-hidden />
          <span className="sr-only">Search ClimateVerse datasets and DOIs</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search datasets, roles or DOIs"
            className="w-full bg-transparent text-[13px] text-paper placeholder:text-faint focus:outline-none"
          />
        </label>
        <Segmented
          size="sm"
          value={onlyIntegrated ? 'used' : 'all'}
          onChange={(value) => setOnlyIntegrated(value === 'used')}
          options={[
            { value: 'used', label: `Integrated (${data.catalog.meta.integrated})`, title: 'Show only the datasets integrated into VayuDrishti' },
            { value: 'all', label: `All ${data.catalog.meta.count}`, title: 'Browse the complete ClimateVerse India catalog' },
          ]}
        />
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 text-[10px] text-faint">
        <span>{rows.length} {rows.length === 1 ? 'dataset' : 'datasets'} shown</span>
        <span>DOI links open the source record</span>
      </div>
      <ul className="mt-1 divide-y divide-line">
        {rows.map((dataset) => (
          <li key={dataset.doi} className="py-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className={cx('text-[13px] font-medium leading-snug', dataset.integrated ? 'text-paper' : 'text-mute')}>
                  {dataset.name}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 text-2xs text-faint">
                  <span>{dataset.curated_category ?? dataset.category}</span>
                  {dataset.integrated && <span className="rounded border border-line px-1.5 py-0.5 font-mono text-[9px] text-mute">INTEGRATED</span>}
                </div>
              </div>
              <a
                href={dataset.url}
                target="_blank"
                rel="noreferrer"
                className="flex shrink-0 items-center gap-1 font-mono text-[10px] text-mute hover:text-paper"
                title={`Open ${dataset.doi}`}
              >
                {dataset.doi.replace('doi:', '')} <ExternalLink size={11} aria-hidden />
              </a>
            </div>
            {dataset.role && <p className="mt-1.5 max-w-[680px] text-xs leading-relaxed text-mute">{dataset.role}</p>}
          </li>
        ))}
        {rows.length === 0 && (
          <li className="py-8 text-center text-[13px] text-mute">
            No datasets match “{query}”. Try a broader word such as “wind” or “school”.
          </li>
        )}
      </ul>
    </div>
  )
}
