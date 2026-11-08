import { useEffect, useRef, useState } from 'react'
import type { Num } from '../../data/types'
import { istParts } from '../../lib/stations'

/** Ward value at an integer hour, or null when the series has no estimate there. */
export function valueAt(series: Num[], i: number): number | null {
  const v = series[i]
  return v != null && Number.isFinite(v) ? v : null
}

/** Highest finite value in [from, from + span), with its index. */
export function peakAhead(series: Num[], from: number, span: number): { value: number; index: number } | null {
  let best: { value: number; index: number } | null = null
  const end = Math.min(series.length, from + span)
  for (let i = Math.max(0, from); i < end; i++) {
    const v = series[i]
    if (v != null && Number.isFinite(v) && (!best || v > best.value)) best = { value: v, index: i }
  }
  return best
}

export function whenLabel(iso: string | undefined): string {
  const p = istParts(iso)
  return iso ? `${p.day} ${p.date} · ${p.time}` : '—'
}

export function fmtInt(n: number | null | undefined): string {
  return n == null || !Number.isFinite(n) ? '—' : Math.round(n).toLocaleString('en-IN')
}

export function fmtMultiple(v: number | null, base: number, digits = 1): string {
  return v == null ? '—' : `${(v / base).toFixed(digits)}×`
}

/** Precomputed directives carry a few inline LaTeX spans; render them as plain code text. */
export function tidyMath(md: string): string {
  return md.replace(/\$([^$\n]+)\$/g, (_, body: string) => {
    const plain = body
      .replace(/\\hat\{\\mu\}/g, 'μ̂')
      .replace(/\\text\{([^}]*)\}/g, '$1')
      .replace(/\\mathbb\{E\}/g, 'E')
      .replace(/\\mid/g, '|')
      .replace(/_\{([^}]*)\}/g, '($1)')
      .replace(/\\/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    return `\`${plain}\``
  })
}

/** Tracks an element's content-box size. */
export function useSize<T extends HTMLElement>(): [React.RefObject<T | null>, { width: number; height: number }] {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize((s) => (Math.abs(s.width - width) < 0.5 && Math.abs(s.height - height) < 0.5 ? s : { width, height }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size]
}
