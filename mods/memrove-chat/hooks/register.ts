import type { EngineInterface, Register } from 'claude-code'
import {
  BACKOFF_MS,
  configFrom,
  failureText,
  freshMessages,
  inBackoff,
  parseApplied,
  parseBatch,
  parseStatus,
  shortenInstruction,
  statusLine,
  viewBlock,
  withoutView,
  type Config,
} from './chat.ts'

/** The mod is transport only: every memory decision lives in the memrove CLI. */
type State = {
  running: boolean
  pending: boolean
  failures: number
  lastFailureAt: number
  nodes?: number
  calls?: number
  cap?: number
  /** The view SessionStart read, for the prompt.context right after it (one read per start). */
  view?: { at: number; read: Promise<string | undefined> }
}

const ZOOM_TOOL = {
  name: 'zoom',
  description:
    'Open line id+n of the chat memory view into the two lines of n/2 under it; n = 1 gives message id whole (a long message comes in pages). n is a power of 2 and id a multiple of n. Use it to find the latest mention of anything that may have come up before, before you act, guess or ask.',
  inputSchema: {
    type: 'object',
    properties: {
      id: { type: ['integer', 'string'], minimum: 0, description: 'The first message of the line (the number before the +); or `rec:<id>` of a saved record from a search hit, which opens the record with its why (n = 1).' },
      n: { type: 'integer', minimum: 1, description: 'How many messages the line covers (the number after the +); a power of 2.' },
      page: { type: 'integer', minimum: 1, description: 'Page of a long message (n = 1 only); default 1.' },
    },
    required: ['id', 'n'],
  },
} as const

const DATE_TOOL = {
  name: 'date',
  description: 'The date and time of message id of the chat memory.',
  inputSchema: { type: 'object', properties: { id: { type: 'integer', minimum: 0 } }, required: ['id'] },
} as const

const SEARCH_TOOL = {
  name: 'search',
  description: 'Search every message of the chat memory, and the saved decisions, lessons and notes, by their words. Returns the best hits as id|date|kind|snippet (a saved record is rec:<id>); a hit is only a snippet that drops detail, so open the message with zoom(id, 1) before you answer from it. Optional since/until limit it to a date range. Use it when the view shows a topic only as a word, or to find where something was said.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'A few words the message would contain. May be empty when since or until is given: then the messages of that range are listed, newest first.' },
      since: { type: 'string', description: 'Only messages from this date on: YYYY-MM-DD or an ISO datetime. Turn words like "last week" into dates yourself.' },
      until: { type: 'string', description: 'Only messages up to this date, inclusive: YYYY-MM-DD or an ISO datetime.' },
    },
    required: ['query'],
  },
} as const

const PROJECT_TOOL = {
  name: 'project',
  description:
    'Where a project stands, in a few lines: its latest decisions (the ones a later decision replaced are marked SUPERSEDED), open loops, and the end state of its last session. Every line starts with an id that zoom opens (rec:<id>, th:<id>, ls:<id>). Call it when you start work on a project and need its current state, before you ask the owner to re-explain. Without a name it uses the project of the working directory.',
  inputSchema: { type: 'object', properties: { name: { type: 'string', description: 'Project name; leave out to use the working directory.' } } },
} as const

const DOC_TOOL = {
  name: 'doc',
  description:
    'Open a docs write-up that a search doc handle points to (the `doc|date|path|heading|snippet` lines under the hits), whole or one section. Only files under docs/state, docs/design, docs/research and docs/decisions of the owner\'s repos open.',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'The absolute path from the doc handle.' },
      section: { type: 'string', description: 'Heading of one section; leave out for the whole file.' },
    },
    required: ['path'],
  },
} as const

const MODEL_TOKENS = 16_000
const MODEL_TIMEOUT_MS = 300_000
const CLI_TIMEOUT_MS = 120_000
const VIEW_HANDOFF_MS = 30_000

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/** Runs the memrove CLI; a non-zero exit throws with the end of its stderr. */
async function cli($: EngineInterface, cfg: Config, args: string[], stdin?: string): Promise<string> {
  const r = await $.process.run([...cfg.memrove, ...args], { timeoutMs: CLI_TIMEOUT_MS, ...(stdin === undefined ? {} : { stdin }) })
  if (r.exitCode !== 0) throw new Error(`memrove ${args.join(' ')} exited ${r.exitCode}: ${(r.stderr || r.stdout).trim().slice(-300)}`)
  return r.stdout
}

function show($: EngineInterface, text: string): void {
  $.ui.status(text)
}

async function refreshStatus($: EngineInterface, state: State, cfg: Config): Promise<void> {
  const s = parseStatus(await cli($, cfg, ['chatlog', 'status']))
  state.nodes = s.nodes
  state.calls = s.calls_today
  state.cap = s.cap
}

/** One batch: prepare, one Haiku call on the session's own auth, apply. Returns true when the queue is empty. */
async function oneBatch($: EngineInterface, state: State, cfg: Config, n: number): Promise<boolean> {
  const batch = parseBatch(await cli($, cfg, ['chatlog', 'prepare', '--json', '--no-ingest']))
  if (batch === undefined) return true
  show($, statusLine({ building: true, batch: n, nodes: state.nodes, calls: state.calls, cap: state.cap }))
  const r = await $.model.complete({ model: cfg.model, system: batch.system, prompt: batch.prompt, maxTokens: MODEL_TOKENS, timeoutMs: MODEL_TIMEOUT_MS })
  if (!r.isAnswered) {
    // Hands the claim back so it does not wait out its expiry; the CLI decides what an error means.
    const why = failureText(r as { reason: string; status?: number | null; error?: string })
    await cli($, cfg, ['chatlog', 'apply', batch.batch_id, '--error', why]).catch(() => undefined)
    throw new Error(`the model gave no text (${why})`)
  }
  const done = parseApplied(await cli($, cfg, ['chatlog', 'apply', batch.batch_id], r.text))
  if (done.error !== undefined) throw new Error(`apply: ${done.error}`)
  return done.done
}

/** Ingest, then up to cfg.batches batches. */
async function run($: EngineInterface, state: State, cfg: Config): Promise<void> {
  show($, statusLine({ building: true, nodes: state.nodes, calls: state.calls, cap: state.cap }))
  await cli($, cfg, ['chatlog', 'ingest'])
  for (let n = 1; n <= cfg.batches; n++) {
    if (await oneBatch($, state, cfg, n)) break
    await refreshStatus($, state, cfg)
  }
  await refreshStatus($, state, cfg)
}

/** One run at a time; a turn that ends during a run asks for one more after it (as memory-save does). */
async function drain($: EngineInterface, state: State, cfg: Config): Promise<void> {
  if (state.running) {
    state.pending = true
    return
  }
  state.running = true
  try {
    do {
      state.pending = false
      if (inBackoff(state.failures, state.lastFailureAt, await $.clock.now())) {
        show($, statusLine({ error: `paused ${Math.round(BACKOFF_MS / 60_000)} min after ${state.failures} failed runs` }))
        break
      }
      try {
        await run($, state, cfg)
        state.failures = 0
        show($, statusLine({ nodes: state.nodes, calls: state.calls, cap: state.cap }))
      } catch (err) {
        state.failures += 1
        state.lastFailureAt = await $.clock.now()
        show($, statusLine({ error: message(err) }))
      }
    } while (state.pending)
  } finally {
    state.running = false
  }
}

/** The whole view as the CLI prints it, or undefined when there is no chat memory yet. */
async function readView($: EngineInterface, cfg: Config): Promise<string | undefined> {
  try {
    const text = (await cli($, cfg, ['wake', '--all'])).trim()
    return text === '' ? undefined : text
  } catch (err) {
    $.ui.log(`memrove-chat: the view was not read: ${message(err)}`, { to: 'debug' })
    return undefined
  }
}

async function toolText($: EngineInterface, cfg: Config, args: string[]): Promise<{ result: string }> {
  const r = await $.process.run([...cfg.memrove, ...args], { timeoutMs: CLI_TIMEOUT_MS })
  return { result: r.exitCode === 0 ? r.stdout.trim() : `error: ${(r.stderr || r.stdout).trim()}` }
}

function intOf(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isInteger(v) ? v : undefined
}

export const register: Register = (on, options) => {
  const cfg = configFrom(options as Record<string, unknown>)
  const state: State = { running: false, pending: false, failures: 0, lastFailureAt: 0 }

  on('session.start', async ($, e, next) => {
    const r = await next(e)
    await $.tool.register(ZOOM_TOOL)
    await $.tool.register(DATE_TOOL)
    await $.tool.register(SEARCH_TOOL)
    await $.tool.register(PROJECT_TOOL)
    await $.tool.register(DOC_TOOL)
    return r
  })

  // A plugin's tool waits behind ToolSearch by default; these are listed, so the model calls them at once.
  on('tool.describe', { tool: /^mcp__memrove-chat__(zoom|date|search|project|doc)$/ }, async (_, e, next) => ({ ...(await next(e)), isDeferred: false }))

  on('tool.call', { tool: /^mcp__memrove-chat__zoom$/ }, async ($, e) => {
    const input = e as Record<string, unknown>
    if (typeof input.id === 'string' && /^(rec|th|ls):[\w:-]+$/.test(input.id.trim())) return toolText($, cfg, ['show', input.id.trim()])
    const id = intOf(input.id)
    const n = intOf(input.n)
    if (id === undefined || n === undefined) return { deny: 'zoom needs integer id and n' }
    const page = intOf(input.page)
    return toolText($, cfg, ['zoom', String(id), String(n), ...(page === undefined ? [] : ['--page', String(page)])])
  })

  on('tool.call', { tool: /^mcp__memrove-chat__date$/ }, async ($, e) => {
    const id = intOf((e as Record<string, unknown>).id)
    return id === undefined ? { deny: 'date needs an integer id' } : toolText($, cfg, ['date', String(id)])
  })

  on('tool.call', { tool: /^mcp__memrove-chat__search$/ }, async ($, e) => {
    const { query, since, until } = e as Record<string, unknown>
    const range = [['--since', since], ['--until', until]].flatMap(([flag, v]) => (typeof v === 'string' && v.trim() !== '' ? [flag as string, v.trim()] : []))
    const text = typeof query === 'string' ? query.trim() : ''
    return text !== '' || range.length > 0 ? toolText($, cfg, ['chatlog', 'search', text, ...range]) : { deny: 'search needs a query or a since/until date' }
  })

  on('tool.call', { tool: /^mcp__memrove-chat__project$/ }, async ($, e) => {
    const name = (e as Record<string, unknown>).name
    return toolText($, cfg, ['project', ...(typeof name === 'string' && name.trim() !== '' ? [name.trim()] : [])])
  })

  on('tool.call', { tool: /^mcp__memrove-chat__doc$/ }, async ($, e) => {
    const { path, section } = e as Record<string, unknown>
    if (typeof path !== 'string' || path.trim() === '') return { deny: 'doc needs a path' }
    return toolText($, cfg, ['doc', path.trim(), ...(typeof section === 'string' && section.trim() !== '' ? ['--section', section.trim()] : [])])
  })

  // The view joins the first user message as a context block of its own: at startup, resume, /clear and after a compaction.
  on('prompt.context', async ($, e, next) => {
    const r = await next(e)
    // Takes the read SessionStart just made (a few seconds old at most); /clear and compaction have none and read again.
    const held = state.view
    state.view = undefined
    const view = await (held !== undefined && (await $.clock.now()) - held.at < VIEW_HANDOFF_MS ? held.read : readView($, cfg))
    return view === undefined ? r : { ...r, blocks: [...r.blocks, { name: 'memroveChat', text: viewBlock(view) }] }
  })

  // Subagents inherit the session's context sections (measured: +33k tokens each); the view stays with the main loop.
  on('prompt.attachment', { type: 'context_sections' }, async (_, e, next) => {
    const r = await next(e)
    if (e.agentId === undefined || r.text === null || r.text === undefined) return r
    return { ...r, text: withoutView(r.text) }
  })

  // The SessionStart instruction of the memrove hook shrinks to one line once the view is in the context.
  on('classic.SessionStart', async ($, e, next) => {
    const r = await next(e)
    if (e.agent_id !== undefined || r.additionalContext === undefined) return r
    // SessionStart runs before prompt.context, so it reads the view itself and hands the same read on to that block.
    const read = readView($, cfg)
    state.view = { at: await $.clock.now(), read }
    if ((await read) === undefined) return r
    return { ...r, additionalContext: shortenInstruction(r.additionalContext) }
  })

  // The fresh context: the view and the last exchange, nothing else. Off unless the `fresh` option is on.
  on('session.compact', async ($, e, next) => {
    if (!cfg.fresh || e.agentId !== undefined) return next(e)
    const view = await readView($, cfg)
    const messages = view === undefined ? undefined : freshMessages(view, e.messages)
    return messages === undefined ? next(e) : { messages }
  })

  on('turn.complete', async ($, e, next) => {
    const r = await next(e)
    if (e.agentId !== undefined || e.reason !== 'answer') return r
    // On a timer, not in the dispatch: a call still in flight when the dispatch closes is aborted.
    $.clock.after(0, () => void drain($, state, cfg))
    if (cfg.fresh) {
      // A compaction started from a timer skips this plugin's own hook, so it starts here, not awaited.
      void $.session.compact().catch((err: unknown) => $.ui.log(`memrove-chat: fresh compaction failed: ${message(err)}`))
    }
    return r
  })
}
