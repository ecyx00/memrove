import { describe, expect, test, tier } from 'claude-code/testing'
import type { SessionMessage } from 'claude-code'
import { configFrom, failureText, freshMessages, inBackoff, parseApplied, parseBatch, shortenInstruction, statusLine, withoutView, BACKOFF_FAILURES, BACKOFF_MS, ONE_LINE } from '../hooks/chat.ts'

tier('user')

const m = (role: 'user' | 'assistant', text: string, extra: Partial<SessionMessage> = {}): SessionMessage => ({ role, text, toolUses: [], ...extra })

describe('chat.ts', () => {
  test('configFrom splits the command and falls back to defaults', () => {
    const c = configFrom({ memrove: 'python3 /a/b.py --vault /v --state /s', batches: 3 })
    expect(c.memrove).toEqual(['python3', '/a/b.py', '--vault', '/v', '--state', '/s'])
    expect(c.fresh).toBe(false)
    const d = configFrom({ memrove: '  ', batches: 99, model: '', fresh: 'x' })
    expect(d).toEqual({ memrove: ['memrove'], batches: 3, model: 'haiku', fresh: false })
    expect(configFrom({ fresh: true }).fresh).toBe(true)
  })

  test('parses the CLI json', () => {
    expect(parseBatch('{}')).toBeUndefined()
    expect(parseBatch('{"batch_id":"a","system":"s","prompt":"p","node_ids":["0+1"]}')?.node_ids).toEqual(['0+1'])
    expect(() => parseBatch('oops')).toThrow('no JSON')
    expect(parseApplied('{"applied":2,"retry_ids":[],"done":true}')).toEqual({ applied: 2, retry_ids: [], done: true })
    expect(parseApplied('{"applied":0,"retry_ids":[],"done":false,"error":"unknown or expired batch"}').error).toBe('unknown or expired batch')
  })

  test('failure text and status line', () => {
    expect(failureText({ reason: 'empty-reply' })).toBe('empty-reply')
    expect(failureText({ reason: 'api-error', status: null, error: 'rate_limit' })).toBe('api-error rate_limit')
    expect(statusLine({ building: true, batch: 2, nodes: 10, calls: 3, cap: 60 })).toBe('chat memory: building batch 2 · 10 nodes · 3/60 calls today')
    expect(statusLine({ error: 'x' })).toBe('chat memory: x')
  })

  test('backoff after repeated failures, lifted after ten minutes', () => {
    expect(inBackoff(BACKOFF_FAILURES - 1, 0, 1)).toBe(false)
    expect(inBackoff(BACKOFF_FAILURES, 1000, 2000)).toBe(true)
    expect(inBackoff(BACKOFF_FAILURES, 1000, 1000 + BACKOFF_MS)).toBe(false)
  })

  test('shortenInstruction swaps the memrove text for one line, once', () => {
    expect(shortenInstruction(['a', 'Memrove chat memory is on: x', 'Memrove chat memory is on: y'])).toEqual(['a', ONE_LINE])
    expect(shortenInstruction(undefined)).toBeUndefined()
  })

  test('freshMessages: from the last user message on, verbatim, without a second copy of the view', () => {
    const msgs = [
      m('user', 'first', { handle: 'h0' }),
      m('assistant', 'a1', { handle: 'h1' }),
      m('user', 'second', { handle: 'h2' }),
      m('assistant', 'using a tool', { handle: 'h3', toolUses: [{ tool_use_id: 't', tool: 'Read', input: {}, text: '' } as never] }),
      m('user', '', { handle: 'h4', toolResults: [{ tool_use_id: 't', text: 'r', isError: false } as never] }),
      m('assistant', 'answer', { handle: 'h5' }),
    ]
    const out = freshMessages('VIEW', msgs)
    expect(out?.map(x => x.role)).toEqual(['user', 'assistant', 'user', 'assistant'])
    expect(out?.[0]?.text).toBe('second')
    expect(out?.map(x => x.handle)).toEqual(['h2', 'h3', 'h4', 'h5'])
  })

  test('freshMessages falls back without a view or a user message', () => {
    expect(freshMessages('', [m('user', 'x')])).toBeUndefined()
    expect(freshMessages('V', [m('assistant', 'x')])).toBeUndefined()
  })

  test('withoutView cuts the view section and keeps the others', () => {
    const view = '# memroveChat\nNOTE\n0+32|a\n32+32|b'
    expect(withoutView(view)).toBeNull()
    expect(withoutView(`# claudeMd\nrules\n${view}`)).toBe('# claudeMd\nrules')
    expect(withoutView(`${view}\n# userEmail\nx@y.z`)).toBe('# userEmail\nx@y.z')
    expect(withoutView(`# claudeMd\nrules\n${view}\n# currentDate\nd`)).toBe('# claudeMd\nrules\n# currentDate\nd')
    expect(withoutView('# claudeMd\nrules')).toBe('# claudeMd\nrules')
  })
})
