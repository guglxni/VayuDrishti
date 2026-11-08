import { useEffect } from 'react'
import { useTwin } from './store'

const HOURS_PER_SECOND = 4

/** Advances the global hour while playing and wires keyboard shortcuts (space, ←/→, shift for ±24 h). */
export function usePlayback() {
  const playing = useTwin((s) => s.playing)

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    let acc = 0
    const tick = (now: number) => {
      acc += ((now - last) / 1000) * HOURS_PER_SECOND
      last = now
      if (acc >= 1) {
        const steps = Math.floor(acc)
        acc -= steps
        const { hour, data, setHour, setPlaying } = useTwin.getState()
        const max = (data?.forecasts.hours.length ?? 1) - 1
        if (hour >= max) setPlaying(false)
        else setHour(hour + steps)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
      const { setHour, setPlaying } = useTwin.getState()
      if (e.code === 'Space') {
        e.preventDefault()
        setPlaying((p) => !p)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        setHour((h) => h + (e.shiftKey ? 24 : 1))
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        setHour((h) => h - (e.shiftKey ? 24 : 1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
