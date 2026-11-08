import { Component, lazy, Suspense, useEffect, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useTwin } from './state/store'
import { usePlayback } from './state/usePlayback'
import { TopBar } from './ui/TopBar'
import { Timeline } from './ui/Timeline'
import { AboutDrawer } from './ui/AboutDrawer'

const MapView = lazy(() => import('./map/MapView').then((m) => ({ default: m.MapView })))
const ColumnView = lazy(() => import('./views/column/ColumnView').then((m) => ({ default: m.ColumnView })))
const ForecastView = lazy(() => import('./views/forecast/ForecastView').then((m) => ({ default: m.ForecastView })))
const WardsView = lazy(() => import('./views/wards/WardsView').then((m) => ({ default: m.WardsView })))

class ViewBoundary extends Component<{ name: string; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="panel max-w-md p-6">
          <div className="text-[15px] font-medium">The {this.props.name} view stopped rendering</div>
          <p className="mt-2 text-sm text-mute">{this.state.error.message}</p>
          <p className="mt-3 text-sm text-mute">The other views still work. Reload the page to try this one again.</p>
        </div>
      </div>
    )
  }
}

function Boot() {
  const progress = useTwin((s) => s.progress)
  const error = useTwin((s) => s.error)
  return (
    <div className="flex h-full items-center justify-center">
      <div className="w-72 text-center">
        <div className="font-deva text-3xl text-paper">वायुदृष्टि</div>
        <div className="mt-1 text-sm text-mute">VayuDrishti Mumbai</div>
        {error ? (
          <p className="mt-6 text-sm text-aqi-verypoor">{error}</p>
        ) : (
          <>
            <div className="mx-auto mt-6 h-px w-full overflow-hidden bg-line">
              <div className="h-full bg-paper transition-all duration-300" style={{ width: `${Math.max(8, progress * 100)}%` }} />
            </div>
            <p className="mt-3 text-2xs text-faint">Loading forecasts, stations and wards</p>
          </>
        )}
      </div>
    </div>
  )
}

export function App() {
  const data = useTwin((s) => s.data)
  const load = useTwin((s) => s.load)
  const view = useTwin((s) => s.view)
  usePlayback()

  useEffect(() => {
    void load()
  }, [load])

  if (!data) return <Boot />

  const showTimeline = view === 'map' || view === 'column'

  return (
    <div className="relative h-full w-full overflow-hidden bg-ink">
      <main className="absolute inset-0">
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            className="absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <ViewBoundary name={view}>
              <Suspense fallback={null}>
                {view === 'map' && <MapView />}
                {view === 'column' && <ColumnView />}
                {view === 'forecast' && <ForecastView />}
                {view === 'wards' && <WardsView />}
              </Suspense>
            </ViewBoundary>
          </motion.div>
        </AnimatePresence>
      </main>
      <TopBar />
      {showTimeline && <Timeline />}
      <AboutDrawer />
    </div>
  )
}
