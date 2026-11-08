import { AlertTriangle, KeyRound, Loader2, Send } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Ward } from '../../data/types'
import { askGemini, probeProxy, storedKey, storeKey } from '../../lib/gemini'
import { geminiPrompt, wardContext } from '../../lib/grap'
import { sampleAt } from '../../lib/series'
import { useData, useTwin } from '../../state/store'
import { Button, cx } from '../../ui/primitives'
import { whenLabel } from './util'

type Route = 'probing' | 'proxy' | 'browser' | 'nokey'

const SUGGESTIONS = [
  'Should schools hold outdoor assembly tomorrow?',
  'Which construction sites need to stop work?',
  'What should civic hospitals prepare for tonight?',
]

interface Answer {
  text: string
  question: string
  ward: string
  when: string
}

function explain(err: unknown, route: Route): string {
  const msg = err instanceof Error ? err.message : String(err)
  if (/api key|API_KEY|permission|unauthori[sz]ed|403|401/i.test(msg)) {
    return route === 'proxy'
      ? 'The server’s Gemini key was rejected. Ask the site administrator to check it, or try again later.'
      : 'Gemini rejected this API key. Check it in Google AI Studio, then save it again below.'
  }
  if (/quota|rate|429|exhausted/i.test(msg)) return 'Gemini’s usage limit was reached. Wait a minute and try again.'
  if (/failed to fetch|network|load failed/i.test(msg)) {
    return 'Could not reach Gemini. Check the connection and try again. The response order above does not need the network.'
  }
  return `Gemini returned an error: ${msg}. Try again, or rephrase the question.`
}

export function Copilot({ ward }: { ward: Ward }) {
  const data = useData()
  const hour = useTwin((s) => s.hour)
  const [route, setRoute] = useState<Route>('probing')
  const [keyDraft, setKeyDraft] = useState('')
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [answer, setAnswer] = useState<Answer | null>(null)
  const reqId = useRef(0)

  useEffect(() => {
    let alive = true
    void probeProxy().then((ok) => {
      if (alive) setRoute(ok ? 'proxy' : storedKey() ? 'browser' : 'nokey')
    })
    return () => {
      alive = false
    }
  }, [])

  const hasEstimate = sampleAt(ward.series, hour) != null
  const ready = route === 'proxy' || route === 'browser'

  async function ask(q: string) {
    const text = q.trim()
    if (!text || !ready || !hasEstimate) return
    const id = ++reqId.current
    const when = whenLabel(data.forecasts.hours[hour])
    setLoading(true)
    setError(null)
    try {
      const { system, prompt } = geminiPrompt(wardContext(data, ward, hour), text)
      const out = await askGemini(route === 'proxy' ? 'proxy' : 'browser', system, prompt, storedKey())
      if (id !== reqId.current) return
      setAnswer({ text: out, question: text, ward: `${ward.code} · ${ward.name}`, when })
    } catch (e) {
      if (id !== reqId.current) return
      setError(explain(e, route))
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }

  function saveKey() {
    const k = keyDraft.trim()
    if (!k) return
    storeKey(k)
    setKeyDraft('')
    setError(null)
    setRoute('browser')
  }

  function forgetKey() {
    storeKey('')
    setRoute(storedKey() ? 'browser' : 'nokey')
  }

  return (
    <section className="panel-solid p-5">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-[15px] font-semibold tracking-tight text-paper">Ask the copilot</h2>
        <span className="text-2xs text-faint">
          {route === 'probing' && 'Checking connection…'}
          {route === 'proxy' && 'Gemini 2.5 Flash · server'}
          {route === 'browser' && (
            <>
              Gemini 2.5 Flash · your key ·{' '}
              <button onClick={forgetKey} className="underline underline-offset-2 hover:text-paper">
                forget
              </button>
            </>
          )}
          {route === 'nokey' && 'Needs a Gemini key'}
        </span>
      </div>
      <p className="mt-1 text-2xs text-mute">
        Answers use this ward’s numbers for {whenLabel(data.forecasts.hours[hour])}. Check them against the response order before acting.
      </p>

      {route === 'nokey' && (
        <form
          className="mt-3 rounded-[10px] border border-line bg-ink p-3"
          onSubmit={(e) => {
            e.preventDefault()
            saveKey()
          }}
        >
          <label className="flex items-center gap-1.5 text-2xs text-mute" htmlFor="wards-gemini-key">
            <KeyRound size={12} /> Gemini API key
          </label>
          <div className="mt-1.5 flex gap-2">
            <input
              id="wards-gemini-key"
              type="password"
              autoComplete="off"
              value={keyDraft}
              onChange={(e) => setKeyDraft(e.target.value)}
              placeholder="Paste key from Google AI Studio"
              className="h-8 min-w-0 flex-1 rounded-[8px] border border-line bg-ink-2 px-2.5 font-mono text-xs text-paper placeholder:font-sans placeholder:text-faint focus:border-line-strong focus:outline-none"
            />
            <Button type="submit" variant="outline" disabled={!keyDraft.trim()}>
              Save
            </Button>
          </div>
          <p className="mt-1.5 text-[10.5px] leading-snug text-faint">
            Kept only for this browser tab. The response order above is computed offline and works without a key.
          </p>
        </form>
      )}

      <form
        className="mt-3"
        onSubmit={(e) => {
          e.preventDefault()
          void ask(question)
        }}
      >
        <label htmlFor="wards-question" className="sr-only">
          Question for the copilot
        </label>
        <textarea
          id="wards-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              void ask(question)
            }
          }}
          rows={3}
          placeholder={`Ask about Ward ${ward.code}, e.g. which schools to alert first`}
          className="block w-full resize-none rounded-[10px] border border-line bg-ink px-3 py-2 text-[13px] leading-relaxed text-paper placeholder:text-faint focus:border-line-strong focus:outline-none"
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-[10.5px] text-faint">
            {!hasEstimate ? 'No ward estimate at this hour. Move the time to ask.' : ready ? '⌘/Ctrl + Enter to send' : ''}
          </span>
          <Button type="submit" variant="primary" disabled={!ready || loading || !question.trim() || !hasEstimate}>
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Send size={13} />}
            {loading ? 'Asking…' : 'Ask'}
          </Button>
        </div>
      </form>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            disabled={loading}
            onClick={() => {
              setQuestion(s)
              void ask(s)
            }}
            className="rounded-full border border-line px-2.5 py-1 text-left text-2xs text-mute transition-colors hover:border-line-strong hover:text-paper disabled:opacity-40"
          >
            {s}
          </button>
        ))}
      </div>

      {error && (
        <div role="alert" className="mt-4 flex gap-2 rounded-[10px] border border-line-strong bg-paper/5 px-3 py-2.5 text-[12.5px] leading-snug text-paper">
          <AlertTriangle size={14} className="mt-0.5 shrink-0 text-mute" />
          <span>
            {error}
            {route === 'browser' && /key/i.test(error) && (
              <button onClick={forgetKey} className="ml-1 underline underline-offset-2">
                Enter a new key
              </button>
            )}
          </span>
        </div>
      )}

      {(answer || loading) && (
        <div className={cx('mt-4 border-t border-line pt-4', loading && 'opacity-50')}>
          {answer && (
            <>
              <div className="mb-2 text-2xs text-faint">
                <span className="text-mute">“{answer.question}”</span> · Ward {answer.ward} · {answer.when}
              </div>
              <div className="prose-doc">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer.text}</ReactMarkdown>
              </div>
            </>
          )}
          {!answer && loading && <div className="text-[13px] text-mute">Drafting an answer for Ward {ward.code}…</div>}
        </div>
      )}
    </section>
  )
}
