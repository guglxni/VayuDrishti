import { createReadStream, existsSync } from 'node:fs'
import os from 'node:os'
import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Guard against macOS Seatbelt / container sandboxes where `uv_interface_addresses`
// throws EPERM (errno 1) when Vite's `resolveServerUrls` enumerates LAN interfaces
// for `host: '0.0.0.0'`.
const rawNetworkInterfaces = os.networkInterfaces.bind(os)
os.networkInterfaces = (() => {
  try {
    return rawNetworkInterfaces()
  } catch {
    return {
      lo0: [
        {
          address: '127.0.0.1',
          netmask: '255.0.0.0',
          family: 'IPv4',
          mac: '00:00:00:00:00:00',
          internal: true,
          cidr: '127.0.0.1/8',
        },
      ],
    }
  }
}) as typeof os.networkInterfaces

const GEMINI_MODEL = 'gemini-2.5-flash'

/**
 * Local API: POST /api/gemini {prompt} -> Gemini 2.5 Flash, using GEMINI_API_KEY from the
 * shell or web/.env.local so the key never ships to the browser bundle.
 */
/**
 * MapLibre 6 locates `maplibre-gl-worker.mjs` next to `import.meta.url`.
 * Vite's dep prebundle moves the library to `.vite/deps/maplibre-gl.js`, so
 * that sibling URL 404s, vector layers never build, and a bad default-import
 * interop can fail the whole module graph (blank page, because CSS is injected
 * by JS). Serve the real worker bytes for any of those URLs.
 */
function maplibreWorkerFallback(): Plugin {
  const workerFile = fileURLToPath(
    new URL('./node_modules/maplibre-gl/dist/maplibre-gl-worker.mjs', import.meta.url),
  )
  return {
    name: 'maplibre-worker-fallback',
    enforce: 'pre',
    configResolved(config) {
      config.server.host = '0.0.0.0'
      config.server.port = 5179
      config.server.strictPort = true
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const raw = req.url ?? ''
        const pathname = raw.split('?')[0]
        const query = raw.includes('?') ? raw.slice(raw.indexOf('?') + 1) : ''
        // `?url` / `?import` must reach Vite so `import x from '...?url'` stays a string.
        if (
          (!pathname.endsWith('/maplibre-gl-worker.mjs') &&
            !pathname.endsWith('/maplibre-gl-worker-dev.mjs')) ||
          /(?:^|&)(?:url|import)(?:&|=|$)/.test(query)
        ) {
          next()
          return
        }
        if (!existsSync(workerFile)) {
          next()
          return
        }
        res.statusCode = 200
        res.setHeader('Content-Type', 'text/javascript; charset=utf-8')
        res.setHeader('Cache-Control', 'no-store')
        createReadStream(workerFile).pipe(res)
      })
    },
  }
}

/**
 * Local API: POST /api/gemini {prompt} -> Gemini 2.5 Flash, using GEMINI_API_KEY from the
 * shell or web/.env.local so the key never ships to the browser bundle.
 */
function geminiProxy(apiKey: string | undefined): Plugin {
  return {
    name: 'vayudrishti-gemini-proxy',
    configureServer(server) {
      server.middlewares.use('/api/gemini', async (req, res) => {
        if (req.method === 'GET') {
          res.setHeader('content-type', 'application/json')
          res.end(JSON.stringify({ available: Boolean(apiKey), model: GEMINI_MODEL }))
          return
        }
        if (!apiKey) {
          res.statusCode = 503
          res.end(JSON.stringify({ error: 'GEMINI_API_KEY is not set for the dev server' }))
          return
        }
        let body = ''
        for await (const chunk of req) body += chunk
        try {
          const { prompt, system } = JSON.parse(body || '{}')
          const upstream = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`,
            {
              method: 'POST',
              headers: { 'content-type': 'application/json', 'x-goog-api-key': apiKey },
              body: JSON.stringify({
                systemInstruction: system ? { parts: [{ text: system }] } : undefined,
                contents: [{ role: 'user', parts: [{ text: prompt }] }],
                generationConfig: { temperature: 0.35, maxOutputTokens: 2048 },
              }),
            },
          )
          res.statusCode = upstream.status
          res.setHeader('content-type', 'application/json')
          res.end(await upstream.text())
        } catch (err) {
          res.statusCode = 500
          res.end(JSON.stringify({ error: String(err) }))
        }
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    base: env.VITE_BASE ?? '/',
    server: {
      host: '0.0.0.0',
      port: 5179,
      strictPort: true,
      cors: true,
    },
    preview: {
      host: '0.0.0.0',
      port: 5179,
      strictPort: true,
      cors: true,
    },
    plugins: [
      maplibreWorkerFallback(),
      react(),
      tailwindcss(),
      geminiProxy(env.GEMINI_API_KEY ?? process.env.GEMINI_API_KEY),
    ],
    optimizeDeps: {
      // MapLibre 6 is ESM with named exports only. Prebundling it makes Vite's
      // CJS interop request a `default` export that does not exist, which
      // aborts the module graph before React paints.
      exclude: ['maplibre-gl'],
    },
    build: {
      target: 'es2022',
      chunkSizeWarningLimit: 1800,
      rollupOptions: {
        output: {
          manualChunks(id: string) {
            if (!id.includes('node_modules')) return undefined
            if (id.includes('maplibre-gl')) return 'maplibre'
            if (/[\\/](@react-three|postprocessing|three-stdlib|troika|camera-controls|maath)[\\/]/.test(id)) return 'r3f'
            if (/[\\/]three[\\/]/.test(id)) return 'three'
            return undefined
          },
        },
      },
    },
  }
})
