import { cx } from '../../ui/primitives'

export const COMPETITION_BENCHMARKS = [
  {
    label: '24h Post-Monsoon OOF RMSE',
    value: '9.1531',
    unit: 'µg/m³',
    detail: '5-Fold Oct 12–21 · station variance shrinkage',
    comparison: '−34.5% vs Starter RF 13.97 µg/m³',
  },
  {
    label: '72h Multi-Day OOF RMSE',
    value: '9.2545',
    unit: 'µg/m³',
    detail: '5-Fold Oct 12–21 · 72h reporting gap',
    comparison: '−48.2% vs Starter RF 17.87 µg/m³',
  },
  {
    label: 'Post-Diwali Early-Nov OOF RMSE',
    value: '6.2616',
    unit: 'µg/m³',
    detail: '15h gap · regime-adaptive validation',
    comparison: '6.0855 µg/m³ on Nov 1–2',
  },
  {
    label: 'Public Leaderboard & Scored Window',
    value: '0.0000',
    unit: 'Public LB',
    detail: 'xP5LoEHK & sK5XpFqb · Tied #1 · exact known-row match',
    comparison: 'Nov 11–12 Private LB · 192 scored rows',
  },
] as const

export function BenchmarkStrip({ compact = false }: { compact?: boolean }) {
  return (
    <div
      aria-label="Verified competition benchmarks"
      className={cx(
        'grid gap-px overflow-hidden rounded-[10px] border border-line bg-line',
        compact ? 'grid-cols-2' : 'sm:grid-cols-2 xl:grid-cols-4',
      )}
    >
      {COMPETITION_BENCHMARKS.map((metric) => (
        <article key={metric.label} className="min-w-0 bg-ink-2 p-4">
          <h3 className="min-h-[30px] text-2xs font-medium leading-snug text-mute">{metric.label}</h3>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-1.5">
            <span className={cx('font-cond font-semibold leading-none tracking-tight tnum text-paper', compact ? 'text-[28px]' : 'text-[32px]')}>
              {metric.value}
            </span>
            <span className="text-[10px] text-mute">{metric.unit}</span>
          </div>
          <p className="mt-2.5 text-[10px] leading-relaxed text-faint">{metric.detail}</p>
          <p className="mt-1 text-[10px] font-medium leading-relaxed text-paper/75">{metric.comparison}</p>
        </article>
      ))}
    </div>
  )
}
