import { describe, expect, mock, test, tier, type Engine, type MockClock, type TestBody } from 'claude-code/testing'
import type { On, SessionMessage, TurnCompleteInput } from 'claude-code'

tier('user')

const usage = { input_tokens: 1, output_tokens: 2, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }
const start = { surface: null, isInteractive: true, cwd: '/src/app' } as const
const turn = (agentId?: string, reason: 'answer' | 'error' | 'aborted' = 'answer'): TurnCompleteInput => ({
  answer: 'done',
  durationMs: 10,
  isAborted: false,
  turnId: 't1',
  reason,
  agentId,
})

type World = {
  clock: MockClock
  /** Every memrove call: the CLI arguments after the entry point, and the stdin. */
  calls: { args: string[]; stdin?: string }[]
  /** Batches the fake CLI still has ready. */
  ready: number
  /** Replies of the model, in order; a string answers, an object is a failed result. */
  replies: (string | { reason: string; status?: number | null; error?: string })[]
  prompts: { model: string; system?: string; prompt: string }[]
  statuses: (string | undefined)[]
  tools: string[]
  view: string
  modelMs: number
  compacts: number
  messages: SessionMessage[]
}

/** The memrove CLI as the mod sees it: [stdout, exit code]. */
function fakeCli(w: World, args: string[]): [string, number] {
  const key = args[0] === 'chatlog' ? `chatlog ${args[1]}` : (args[0] ?? '')
  const table: Record<string, () => [string, number]> = {
    'chatlog ingest': () => ['{}', 0],
    'chatlog prepare': () => {
      if (w.ready === 0) return ['{}', 0]
      w.ready -= 1
      return [JSON.stringify({ batch_id: `b${w.calls.length}`, system: 'SYS', prompt: 'PROMPT', node_ids: ['0+1'] }), 0]
    },
    'chatlog apply': () => [JSON.stringify({ applied: 1, retry_ids: [], done: w.ready === 0 }), 0],
    'chatlog status': () => [JSON.stringify({ nodes: 7, calls_today: 2, cap: 60, messages: 4 }), 0],
    wake: () => (w.view === '' ? ['no chat memory yet', 1] : [w.view, 0]),
    zoom: () => (args[1] === '3' ? ['no line 3+8', 1] : [`zoomed ${args.slice(1).join(' ')}`, 0]),
    'chatlog search': () => [`7|2026-10-04 10:00|user|hit for ${args[2]}`, 0],
    show: () => [`record ${args[1]}`, 0],
    date: () => ['2026-10-04 10:00:00 +0300', 0],
    project: () => [`project ${args[1] ?? 'cwd'}`, 0],
    doc: () => [`doc ${args.slice(1).join(' ')}`, 0],
  }
  return (table[key] ?? (() => ['', 2]))()
}

function world(on: On, sessionStart: string[] = []): World {
  const w: World = { clock: mock.clock(on, { now: Date.parse('2026-10-08T12:00:00Z') }), calls: [], ready: 0, replies: [], prompts: [], statuses: [], tools: [], view: '', modelMs: 1000, compacts: 0, messages: [] }
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('turn.complete', (_, e) => ({ text: e.answer }))
  on('tool.register', (_, e) => { w.tools.push(e.name); return { value: { tool: `mcp__memrove-chat__${e.name}` } } })
  on('tool.call', () => ({ result: undefined, text: '' }))
  on('prompt.context', (_, e) => ({ blocks: e.blocks }))
  on('prompt.attachment', (_, e) => ({ text: e.text }))
  on('classic.SessionStart', () => ({ additionalContext: sessionStart }))
  on('session.messages', () => ({ value: w.messages }))
  on('session.compact', () => { w.compacts += 1; return { messages: [{ role: 'user', text: 'engine summary', toolUses: [] }] } })
  on('ui.status', (_, e) => { w.statuses.push(e.text); return { value: undefined } })
  on('ui.log', () => ({ value: undefined }))
  on('process.run', (_, e) => {
    const args = e.argv.slice(2) // the test's entry point is `fake memrove`
    w.calls.push({ args, ...(e.init?.stdin === undefined ? {} : { stdin: e.init.stdin }) })
    const [stdout, exitCode] = fakeCli(w, args)
    return { value: { exitCode, stdout, stderr: exitCode === 0 ? '' : stdout, isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('model.complete', async (_, e) => {
    w.prompts.push({ model: e.model, prompt: e.prompt, ...(e.system === undefined ? {} : { system: e.system }) })
    await w.clock.sleep(w.modelMs)
    const r = w.replies.shift() ?? 'default'
    return { value: typeof r === 'string' ? { isAnswered: true, text: r, usage } : { isAnswered: false, usage, ...r } } as never
  })
  return w
}

const OPTIONS = { memrove: 'fake memrove', batches: 3 }
const it = (name: string, body: TestBody) => test(name, { options: OPTIONS }, body)
const fresh = (name: string, body: TestBody) => test(name, { options: { ...OPTIONS, fresh: true } }, body)

const verbs = (w: World): string[] => w.calls.map(c => c.args.slice(0, 2).join(' '))

async function ended($: Engine, w: World, t = turn()): Promise<void> {
  await $.turn.complete(t)
  await run(w)
}

/** The run is unawaited and each model call takes a second of the mock clock, so the test moves the clock past them. */
async function run(w: World): Promise<void> {
  for (let i = 0; i < 8; i++) await w.clock.advance(w.modelMs)
  await w.clock.settle()
}

describe('memrove-chat', () => {
  it('registers zoom, date, search, project and doc', async ($, on) => {
    const w = world(on)
    await $.session.start(start)
    expect(w.tools).toEqual(['zoom', 'date', 'search', 'project', 'doc'])
  })

  it('zoom and date call the CLI and return its text', async ($, on) => {
    world(on)
    await $.session.start(start)
    const z = await $.tool.call({ tool: 'mcp__memrove-chat__zoom', id: 40, n: 8 })
    expect(JSON.stringify(z)).toContain('zoomed 40 8')
    const p = await $.tool.call({ tool: 'mcp__memrove-chat__zoom', id: 5, n: 1, page: 2 })
    expect(JSON.stringify(p)).toContain('zoomed 5 1 --page 2')
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__date', id: 5 }))).toContain('2026-10-04')
  })

  it('zoom opens a rec: hit through show', async ($, on) => {
    world(on)
    await $.session.start(start)
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__zoom', id: 'rec:obs-abc', n: 1 }))).toContain('record rec:obs-abc')
  })

  it('project calls the CLI with the name or none, and zoom opens th: and ls: ids through show', async ($, on) => {
    world(on)
    await $.session.start(start)
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__project', name: 'alpha' }))).toContain('project alpha')
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__project' }))).toContain('project cwd')
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__zoom', id: 'th:t-ab12', n: 1 }))).toContain('record th:t-ab12')
  })

  it('doc passes the path and section to the CLI and denies an empty path', async ($, on) => {
    world(on)
    await $.session.start(start)
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__doc', path: '/r/docs/state/a.md', section: 'Result' }))).toContain('doc /r/docs/state/a.md --section Result')
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__doc', path: ' ' }))).toContain('doc needs a path')
  })

  it('search passes the query to the CLI and denies an empty one', async ($, on) => {
    world(on)
    await $.session.start(start)
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__search', query: 'kararlar' }))).toContain('7|2026-10-04 10:00|user|hit for kararlar')
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__search', query: ' ' }))).toContain('search needs a query')
  })

  it('returns the CLI error text of a bad zoom, and denies a call without numbers', async ($, on) => {
    world(on)
    await $.session.start(start)
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__zoom', id: 3, n: 8 }))).toContain('error: no line 3+8')
    expect(JSON.stringify(await $.tool.call({ tool: 'mcp__memrove-chat__zoom', id: 'x', n: 8 }))).toContain('zoom needs integer id and n')
  })

  it('ingests first, then runs batches through the model until the cap of 3', async ($, on) => {
    const w = world(on)
    w.ready = 10
    w.replies.push('[[1]] a', '[[1]] b', '[[1]] c', '[[1]] never')
    await $.session.start(start)
    await ended($, w)
    const v = verbs(w)
    expect(v[0]).toBe('chatlog ingest')
    expect(v.filter(x => x === 'chatlog prepare')).toHaveLength(3)
    expect(w.prompts).toHaveLength(3)
    expect(w.prompts[0]).toEqual({ model: 'haiku', system: 'SYS', prompt: 'PROMPT' })
    const applies = w.calls.filter(c => c.args[1] === 'apply')
    expect(applies.map(c => c.stdin)).toEqual(['[[1]] a', '[[1]] b', '[[1]] c'])
    expect(w.calls.find(c => c.args[1] === 'prepare')?.args).toContain('--no-ingest')
    expect(w.statuses.at(-1)).toBe('chat memory: idle · 7 nodes · 2/60 calls today')
    expect(w.statuses.some(s => s?.startsWith('chat memory: building'))).toBe(true)
  })

  it('stops when the CLI says done', async ($, on) => {
    const w = world(on)
    w.ready = 1
    await $.session.start(start)
    await ended($, w)
    expect(w.prompts).toHaveLength(1)
    expect(verbs(w).filter(x => x === 'chatlog prepare')).toHaveLength(1)
  })

  it('skips subagent turns and turns that did not end in an answer', async ($, on) => {
    const w = world(on)
    w.ready = 5
    await $.session.start(start)
    await ended($, w, turn('agent-1'))
    await ended($, w, turn(undefined, 'error'))
    await ended($, w, turn(undefined, 'aborted'))
    expect(w.calls).toHaveLength(0)
  })

  it('hands the claim back with apply --error when the model fails, and stops the run', async ($, on) => {
    const w = world(on)
    w.ready = 5
    w.replies.push({ reason: 'api-error', status: 529, error: 'overloaded' })
    await $.session.start(start)
    await ended($, w)
    const errs = w.calls.filter(c => c.args.includes('--error'))
    expect(errs).toHaveLength(1)
    expect(errs[0]?.args.at(-1)).toBe('api-error 529 overloaded')
    expect(w.prompts).toHaveLength(1)
    expect(w.statuses.at(-1)).toBe('chat memory: the model gave no text (api-error 529 overloaded)')
  })

  it('a turn that ends during a run asks for one more run after it', async ($, on) => {
    const w = world(on)
    w.ready = 1
    await $.session.start(start)
    await $.turn.complete(turn())
    await w.clock.advance(10) // the first run is inside the model call
    w.ready = 1
    await $.turn.complete(turn())
    await run(w)
    expect(verbs(w).filter(x => x === 'chatlog ingest')).toHaveLength(2)
    expect(w.prompts).toHaveLength(2)
  })

  it('adds the view as a context block, and leaves the blocks alone without a chat memory', async ($, on) => {
    const w = world(on)
    w.view = '<chat part="1/1">\n0+1|user: hi\n</chat>'
    await $.session.start(start)
    const r = await $.prompt.context({ blocks: [{ name: 'claudeMd', text: 'rules' }] })
    expect(r.blocks.map(b => b.name)).toEqual(['claudeMd', 'memroveChat'])
    expect(r.blocks[1]?.text).toContain('0+1|user: hi')
    w.view = ''
    expect((await $.prompt.context({ blocks: [{ name: 'claudeMd', text: 'rules' }] })).blocks).toHaveLength(1)
  })

  it('reads the view once per start: SessionStart hands its read to the prompt.context after it', async ($, on) => {
    const w = world(on, ['Memrove chat memory is on: long text'])
    w.view = 'VIEW'
    await $.session.start(start)
    await $.classic.SessionStart({ source: 'startup', session_id: 's' } as never)
    const r = await $.prompt.context({ blocks: [] })
    expect(r.blocks[0]?.text).toContain('VIEW')
    expect(w.calls.filter(c => c.args[0] === 'wake')).toHaveLength(1)
    await $.prompt.context({ blocks: [] }) // /clear or a compaction: reads again
    expect(w.calls.filter(c => c.args[0] === 'wake')).toHaveLength(2)
  })

  it('keeps the view out of a subagent: its context_sections lose the memroveChat section only', async ($, on) => {
    world(on)
    await $.session.start(start)
    const text = '# claudeMd\nrules\n# memroveChat\nNOTE\n0+32|a\n# currentDate\nd'
    const sub = await $.prompt.attachment({ type: 'context_sections', text, agentId: 'a1' } as never)
    expect(sub.text).toBe('# claudeMd\nrules\n# currentDate\nd')
    const main = await $.prompt.attachment({ type: 'context_sections', text } as never)
    expect(main.text).toBe(text)
  })

  it('shrinks the memrove hook instruction to one line when the view is in the context', async ($, on) => {
    const w = world(on, ['Memrove chat memory is on: long text\nFirst run `memrove wake`', 'other'])
    w.view = 'VIEW'
    await $.session.start(start)
    const r = await $.classic.SessionStart({ source: 'startup', session_id: 's' } as never)
    expect(r.additionalContext).toHaveLength(2)
    expect(r.additionalContext?.[0]).toMatch(/^Before you answer or start a task that may build on earlier work, search first;/)
    expect(r.additionalContext?.[1]).toBe('other')
  })

  it('keeps the whole instruction when the view could not be added', async ($, on) => {
    const w = world(on, ['Memrove chat memory is on: long text'])
    w.view = ''
    await $.session.start(start)
    const r = await $.classic.SessionStart({ source: 'startup', session_id: 's' } as never)
    expect(r.additionalContext).toEqual(['Memrove chat memory is on: long text'])
  })

  it('fresh is off by default: no compaction after the turn, the engine compacts as usual', async ($, on) => {
    const w = world(on)
    w.view = 'VIEW'
    await $.session.start(start)
    await ended($, w)
    expect(w.compacts).toBe(0)
    const msgs: SessionMessage[] = [{ role: 'user', text: 'hi', toolUses: [], handle: 'h0' }]
    const r = await $.session.compact({ trigger: 'manual', messages: msgs })
    expect(r.messages?.[0]?.text).toBe('engine summary')
    expect(w.compacts).toBe(1)
  })

  fresh('fresh on: the compaction returns the view plus the last exchange, and a turn starts one', async ($, on) => {
    const w = world(on)
    w.view = 'VIEW'
    await $.session.start(start)
    const msgs: SessionMessage[] = [
      { role: 'user', text: 'old', toolUses: [], handle: 'h0' },
      { role: 'assistant', text: 'old answer', toolUses: [], handle: 'h1' },
      { role: 'user', text: 'now', toolUses: [], handle: 'h2' },
      { role: 'assistant', text: 'new answer', toolUses: [], handle: 'h3' },
    ]
    const r = await $.session.compact({ trigger: 'manual', messages: msgs })
    expect(r.messages?.map(m => m.role)).toEqual(['user', 'assistant'])
    expect(r.messages?.[0]?.text).toBe('now')
    expect(w.compacts).toBe(0) // the engine's own compaction did not run
    await ended($, w)
    expect(w.compacts).toBe(1) // the turn asked for the compaction; no messages to keep, so the engine's ran
  })
})
