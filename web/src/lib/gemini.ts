const MODEL = 'gemini-2.5-flash'
const KEY_STORAGE = 'vayudrishti.gemini.key'

export type GeminiRoute = 'proxy' | 'browser' | 'offline'

export async function probeProxy(): Promise<boolean> {
  try {
    const r = await fetch('/api/gemini', { method: 'GET' })
    if (!r.ok) return false
    const j = await r.json()
    return Boolean(j.available)
  } catch {
    return false
  }
}

export function storedKey(): string {
  return sessionStorage.getItem(KEY_STORAGE) ?? import.meta.env.VITE_GEMINI_API_KEY ?? ''
}

export function storeKey(k: string) {
  if (k) sessionStorage.setItem(KEY_STORAGE, k)
  else sessionStorage.removeItem(KEY_STORAGE)
}

function extractText(json: unknown): string {
  const j = json as { candidates?: { content?: { parts?: { text?: string }[] } }[]; error?: { message?: string } }
  if (j.error?.message) throw new Error(j.error.message)
  const text = j.candidates?.[0]?.content?.parts?.map((p) => p.text ?? '').join('') ?? ''
  if (!text) throw new Error('Gemini returned an empty response')
  return text
}

export async function askGemini(route: GeminiRoute, system: string, prompt: string, key: string): Promise<string> {
  if (route === 'proxy') {
    const r = await fetch('/api/gemini', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ system, prompt }),
    })
    return extractText(await r.json())
  }
  if (route === 'browser') {
    if (!key) throw new Error('Add a Gemini API key first')
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.35, maxOutputTokens: 2048 },
      }),
    })
    return extractText(await r.json())
  }
  throw new Error('Offline route has no remote model')
}
