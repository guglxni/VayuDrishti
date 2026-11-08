import { Check, Copy } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Ward } from '../../data/types'
import { offlineDirective, wardContext } from '../../lib/grap'
import { fmt, sampleAt } from '../../lib/series'
import { useData, useTwin } from '../../state/store'
import { Button, Segmented } from '../../ui/primitives'
import { GrapChip } from './parts'
import { tidyMath, whenLabel } from './util'

type Mode = 'hour' | 'scored'

export function ResponseOrder({ ward }: { ward: Ward }) {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const setHour = useTwin((s) => s.setHour)
  const [mode, setMode] = useState<Mode>('hour')
  const [copied, setCopied] = useState(false)

  const hasEstimate = sampleAt(ward.series, hour) != null
  const ctx = useMemo(() => (hasEstimate ? wardContext(data, ward, hour) : null), [data, ward, hour, hasEstimate])
  const hourMd = useMemo(() => (ctx ? offlineDirective(ctx) : ''), [ctx])
  const scoredMd = useMemo(() => tidyMath(ward.directive_md), [ward.directive_md])
  const md = mode === 'hour' ? hourMd : scoredMd

  // Same trigger rule as offlineDirective: the higher of now and 85% of the 24-h peak.
  const trigger = ctx ? Math.max(ctx.pm, ctx.peak24 * 0.85) : null
  const peakIdx = data.forecasts.hours.indexOf(ward.private_peak_at)

  useEffect(() => setCopied(false), [md])

  async function copy() {
    try {
      await navigator.clipboard.writeText(md)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  return (
    <section className="panel-solid p-5">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-[15px] font-semibold tracking-tight text-paper">Response order</h2>
        <span className="font-mono text-2xs text-faint">Ward {ward.code}</span>
      </div>

      <Segmented<Mode>
        className="mt-3 w-full [&>button]:flex-1"
        size="sm"
        value={mode}
        onChange={setMode}
        options={[
          { value: 'hour', label: 'This hour', title: 'Recomputed from the ward forecast and city weather for the selected hour' },
          { value: 'scored', label: 'Scored-window order', title: 'Precomputed order for the ward’s peak in the 11–12 Nov scored window' },
        ]}
      />

      <div className="mt-3 flex items-start justify-between gap-3 border-b border-line pb-3">
        <div className="min-w-0 text-2xs leading-snug text-mute">
          {mode === 'hour' ? (
            <>
              For <span className="text-paper">{whenLabel(data.forecasts.hours[hour])}</span>. Trigger uses the higher of PM2.5 now and 85% of the
              24-h peak{trigger != null && <> ({fmt(trigger, 0)} µg/m³)</>}.
            </>
          ) : (
            <>
              For the ward’s scored-window peak,{' '}
              {peakIdx >= 0 ? (
                <button className="text-paper underline decoration-line-strong underline-offset-2 hover:decoration-paper" onClick={() => setHour(peakIdx)}>
                  {whenLabel(ward.private_peak_at)}
                </button>
              ) : (
                <span className="text-paper">{whenLabel(ward.private_peak_at)}</span>
              )}{' '}
              · peak {fmt(ward.private_peak, 0)} µg/m³, window mean {fmt(ward.private_mean, 0)}.
            </>
          )}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          <GrapChip value={mode === 'hour' ? trigger : ward.private_peak} />
          {md && (
            <Button className="!px-1.5 !py-0.5 !text-2xs" onClick={() => void copy()} title="Copy the order as Markdown">
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </Button>
          )}
        </div>
      </div>

      {md ? (
        <div className="prose-doc mt-3 [&_h3:first-child]:hidden">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown>
        </div>
      ) : (
        <div className="mt-4 text-[13px] leading-relaxed text-mute">
          <p>There is no ward estimate for this hour, so no order is issued.</p>
          <p className="mt-2">
            Move the time to a forecast hour, or switch to the{' '}
            <button className="text-paper underline underline-offset-2" onClick={() => setMode('scored')}>
              scored-window order
            </button>
            .
          </p>
        </div>
      )}
    </section>
  )
}
