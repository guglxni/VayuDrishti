import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url'
import { setWorkerUrl } from 'maplibre-gl'

// Must run before the first `new Map()`. MapLibre otherwise derives the worker
// URL from its own module URL, which is wrong once Vite relocates the package.
// Guard against any cached non-string module namespace so `url.endsWith(...)`
// inside MapLibre's worker pool can never throw a TypeError.
const fallbackWorkerUrl = `${import.meta.env.BASE_URL || '/'}maplibre-gl-worker.mjs`
const resolvedWorkerUrl =
  typeof workerUrl === 'string' && workerUrl.length > 0
    ? workerUrl
    : fallbackWorkerUrl

setWorkerUrl(resolvedWorkerUrl)

