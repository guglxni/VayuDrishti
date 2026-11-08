import type { ReactNode } from 'react'
import { pmBand, pmColor } from '../lib/aqi'
import { fmt } from '../lib/series'

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(' ')
}

export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx('panel', className)}>{children}</div>
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx('eyebrow', className)}>{children}</div>
}

/** Large PM2.5 readout in condensed numerals, coloured by CPCB band. */
export function PmValue({
  value,
  size = 'lg',
  unit = true,
  colored = true,
}: {
  value: number | null | undefined
  size?: 'sm' | 'md' | 'lg' | 'xl'
  unit?: boolean
  colored?: boolean
}) {
  const cls = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-[44px] leading-none',
    xl: 'text-[64px] leading-none',
  }[size]
  return (
    <span className="inline-flex items-baseline gap-1.5">
      <span
        className={cx('font-cond font-semibold tnum tracking-tight', cls)}
        style={colored ? { color: pmColor(value) } : undefined}
      >
        {fmt(value, size === 'sm' ? 1 : 0)}
      </span>
      {unit && <span className="text-xs text-mute">µg/m³</span>}
    </span>
  )
}

export function AqiChip({ value, className }: { value: number | null | undefined; className?: string }) {
  const b = pmBand(value)
  return (
    <span
      className={cx('inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-2xs font-medium', className)}
      style={{ background: `${b.color}1f`, color: b.color, boxShadow: `inset 0 0 0 1px ${b.color}40` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: b.color }} />
      {b.label}
    </span>
  )
}

export function Stat({
  label,
  value,
  unit,
  hint,
  className,
}: {
  label: string
  value: ReactNode
  unit?: string
  hint?: ReactNode
  className?: string
}) {
  return (
    <div className={cx('min-w-0', className)}>
      <div className="text-2xs text-faint">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="font-cond text-lg font-medium tnum text-paper">{value}</span>
        {unit && <span className="text-2xs text-mute">{unit}</span>}
      </div>
      {hint && <div className="text-2xs text-mute">{hint}</div>}
    </div>
  )
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  size = 'md',
  className,
}: {
  options: { value: T; label: ReactNode; title?: string }[]
  value: T
  onChange: (v: T) => void
  size?: 'sm' | 'md'
  className?: string
}) {
  return (
    <div
      role="tablist"
      className={cx('inline-flex rounded-[10px] border border-line bg-ink/60 p-0.5', className)}
    >
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={String(o.value)}
            role="tab"
            aria-selected={active}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={cx(
              'rounded-[8px] font-medium transition-colors',
              size === 'sm' ? 'px-2.5 py-1 text-2xs' : 'px-3.5 py-1.5 text-[13px]',
              active ? 'bg-paper text-ink' : 'text-mute hover:text-paper',
            )}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  swatch,
}: {
  checked: boolean
  onChange: () => void
  label: string
  hint?: string
  swatch?: ReactNode
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className="group flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left hover:bg-paper/5"
    >
      <span
        className={cx(
          'relative h-4 w-7 shrink-0 rounded-full transition-colors',
          checked ? 'bg-paper' : 'bg-paper/15',
        )}
      >
        <span
          className={cx(
            'absolute top-0.5 h-3 w-3 rounded-full transition-all',
            checked ? 'left-3.5 bg-ink' : 'left-0.5 bg-paper/60',
          )}
        />
      </span>
      <span className="min-w-0 flex-1">
        <span className={cx('block text-[13px]', checked ? 'text-paper' : 'text-mute')}>{label}</span>
        {hint && <span className="block text-2xs text-faint">{hint}</span>}
      </span>
      {swatch}
    </button>
  )
}

export function Button({
  children,
  onClick,
  variant = 'ghost',
  className,
  title,
  disabled,
  type = 'button',
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'ghost' | 'outline'
  className?: string
  title?: string
  disabled?: boolean
  type?: 'button' | 'submit'
}) {
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cx(
        'inline-flex items-center justify-center gap-1.5 rounded-[10px] px-3 py-1.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'primary' && 'bg-paper text-ink hover:bg-white',
        variant === 'ghost' && 'text-mute hover:bg-paper/8 hover:text-paper',
        variant === 'outline' && 'border border-line-strong text-paper hover:bg-paper/8',
        className,
      )}
    >
      {children}
    </button>
  )
}

/** CPCB NAQI legend strip. */
export function AqiLegend({ compact = false }: { compact?: boolean }) {
  const bands = [
    { label: 'Good', max: '30', color: '#3DAE6B' },
    { label: 'Satisfactory', max: '60', color: '#A7CF4E' },
    { label: 'Moderate', max: '90', color: '#EFD143' },
    { label: 'Poor', max: '120', color: '#F0973A' },
    { label: 'Very poor', max: '250', color: '#E2483F' },
    { label: 'Severe', max: '', color: '#B02A41' },
  ]
  return (
    <div>
      <div className="flex h-1.5 overflow-hidden rounded-full">
        {bands.map((b) => (
          <div key={b.label} className="flex-1" style={{ background: b.color }} />
        ))}
      </div>
      <div className="mt-1 flex text-[10px] text-faint tnum">
        {bands.map((b) => (
          <div key={b.label} className="flex-1 text-right">
            {b.max}
          </div>
        ))}
      </div>
      {!compact && (
        <div className="mt-0.5 flex text-[10px] text-mute">
          <div className="flex-1">Good</div>
          <div className="flex-1 text-center">Moderate</div>
          <div className="flex-1 text-right">Severe</div>
        </div>
      )}
    </div>
  )
}
