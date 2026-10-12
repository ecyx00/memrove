/**
 * The pure part of memrove-chat: the options, what the CLI prints, the status line and the message
 * list of a fresh context. No `$` here, so the tests need no engine.
 */
import type { SessionMessage } from 'claude-code'

export type Config = {
  /** argv of the memrove entry point, split on spaces. */
  memrove: string[]
  /** Batches (one model call each) after a turn. */
  batches: number
  model: string
  /** Compact to the view plus the last exchange after every turn. */
  fresh: boolean
}

export const DEFAULTS = { memrove: 'memrove', batches: 3, model: 'haiku', fresh: false }

export function splitCommand(text: string): string[] {
  return text.split(/\s+/).filter(p => p !== '')
}

export function configFrom(options: Record<string, unknown>): Config {
  const memrove = typeof options.memrove === 'string' && options.memrove.trim() !== '' ? options.memrove : DEFAULTS.memrove
  const b = options.batches
  const batches = typeof b === 'number' && Number.isInteger(b) && b >= 1 && b <= 20 ? b : DEFAULTS.batches
  const model = typeof options.model === 'string' && options.model.trim() !== '' ? options.model.trim() : DEFAULTS.model
  return { memrove: splitCommand(memrove), batches, model, fresh: options.fresh === true || options.fresh === 'true' }
}

export type Batch = { batch_id: string; system: string; prompt: string; node_ids: string[] }
export type Applied = { applied: number; retry_ids: string[]; done: boolean; error?: string }
export type Status = { nodes: number; calls_today: number; cap: number; messages: number }

function parseObject(text: string): Record<string, unknown> | undefined {
  try {
    const v: unknown = JSON.parse(text)
    return typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined
  } catch {
    return undefined
  }
}

/** `chatlog prepare --json`: a batch, or undefined for `{}` (nothing ready, locked, or the cap is spent). */
export function parseBatch(text: string): Batch | undefined {
  const o = parseObject(text)
  if (o === undefined) throw new Error(`prepare printed no JSON: ${text.slice(0, 120)}`)
  if (typeof o.batch_id !== 'string') return undefined
  if (typeof o.system !== 'string' || typeof o.prompt !== 'string') throw new Error('prepare printed a batch without system or prompt')
  return { batch_id: o.batch_id, system: o.system, prompt: o.prompt, node_ids: Array.isArray(o.node_ids) ? o.node_ids.map(String) : [] }
}

export function parseApplied(text: string): Applied {
  const o = parseObject(text)
  if (o === undefined || typeof o.applied !== 'number') throw new Error(`apply printed no result: ${text.slice(0, 120)}`)
  return {
    applied: o.applied,
    retry_ids: Array.isArray(o.retry_ids) ? o.retry_ids.map(String) : [],
    done: o.done === true,
    ...(typeof o.error === 'string' ? { error: o.error } : {}),
  }
}

export function parseStatus(text: string): Status {
  const o = parseObject(text)
  const n = (k: string): number => (typeof o?.[k] === 'number' ? o[k] : 0)
  if (o === undefined) throw new Error(`status printed no JSON: ${text.slice(0, 120)}`)
  return { nodes: n('nodes'), calls_today: n('calls_today'), cap: n('cap'), messages: n('messages') }
}

/** Why a model call gave no text, as one short phrase for `apply --error` and the status line. */
export function failureText(r: { reason: string; status?: number | null; error?: string }): string {
  if (r.reason !== 'api-error') return r.reason
  return `api-error${r.status == null ? '' : ` ${r.status}`}${r.error === undefined ? '' : ` ${r.error}`}`
}

/** The status line: building, nodes done and calls made today, or the error. */
export function statusLine(s: { building?: boolean; nodes?: number; calls?: number; cap?: number; error?: string; batch?: number }): string {
  if (s.error !== undefined) return `chat memory: ${s.error}`
  const parts = [s.building ? `building${s.batch === undefined ? '' : ` batch ${s.batch}`}` : 'idle']
  if (s.nodes !== undefined) parts.push(`${s.nodes} nodes`)
  if (s.calls !== undefined) parts.push(`${s.calls}${s.cap === undefined ? '' : `/${s.cap}`} calls today`)
  return `chat memory: ${parts.join(' · ')}`
}

/** After a run of failures the turns stop asking for a batch for a while, so a poisoned node does not cost 3 calls a turn. */
export const BACKOFF_FAILURES = 3
export const BACKOFF_MS = 10 * 60_000

export function inBackoff(failures: number, lastFailureAt: number, now: number): boolean {
  return failures >= BACKOFF_FAILURES && now - lastFailureAt < BACKOFF_MS
}

/** What the classic SessionStart instruction becomes when the view itself is in the context. */
export const ONE_LINE = 'Before you answer or start a task that may build on earlier work, search first; a hit or a doc line is a lead, not evidence; open it with zoom or memrove doc and answer from what you opened. On one topic the newer statement wins, and the user\'s word beats an agent\'s. Tools: search, zoom (id, n), date.'
const INSTRUCTION_MARK = 'Memrove chat memory is on'

export function shortenInstruction(texts: readonly string[] | undefined): string[] | undefined {
  if (texts === undefined) return undefined
  const out: string[] = []
  for (const t of texts) {
    if (!t.includes(INSTRUCTION_MARK)) out.push(t)
    else if (!out.includes(ONE_LINE)) out.push(ONE_LINE)
  }
  return out
}

export const VIEW_NOTE = 'This is an index of the whole chat so far, one line per run of messages (id+n|summary), not evidence: the newest lines are the most exact, older ones keep little. Open a line with the zoom tool before you rely on it.'

/** The block text of the view. */
export function viewBlock(view: string): string {
  return `${VIEW_NOTE}\n${view}`
}

/**
 * The text of a `context_sections` attachment without the view: subagents get the context blocks of the
 * session, and the view (about 32k tokens) is for the main loop only. Sections are `# name` headers; the
 * view's lines start with a number, so the next core header ends it. Empty text comes back as null.
 */
export function withoutView(text: string): string | null {
  const head = '# memroveChat\n'
  const at = text.startsWith(head) ? 0 : text.indexOf(`\n${head}`) + 1
  if (at === 0 && !text.startsWith(head)) return text
  const end = text.slice(at).search(/\n# (claudeMd|userEmail|attachedProject|currentDate)\n/)
  const rest = end < 0 ? text.slice(0, at).trimEnd() : text.slice(0, at) + text.slice(at + end + 1)
  return rest.trim() === '' ? null : rest
}

/**
 * The messages of a fresh context: the last user message and everything after it verbatim (its tool
 * calls and answer); the view comes back as the prompt.context block. Undefined when there is no view or
 * user message with text, so the compaction falls back to the engine's.
 */
export function freshMessages(view: string, messages: readonly SessionMessage[]): SessionMessage[] | undefined {
  let last = -1
  messages.forEach((m, i) => {
    if (m.role === 'user' && m.text.trim() !== '' && (m.toolResults ?? []).length === 0) last = i
  })
  const user = messages[last]
  if (user === undefined || view.trim() === '') return undefined
  // The view itself is not put in: prompt.context adds it again after every compaction (measured: both copies cost ~2x the view per turn).
  return messages.slice(last)
}
