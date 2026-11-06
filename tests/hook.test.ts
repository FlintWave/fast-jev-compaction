import { describe, expect, it, vi } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  jevAsker,
  register,
  SYSTEMONE_MAX_BODY_BYTES,
  SYSTEMONE_MAX_REQUEST_TOKENS,
  SYSTEMONE_MAX_STATE_TOKENS,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/metzger.ts';
import { ChatJevAsker } from '../src/chat-asker.js';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({ baseUrl: 'http://localhost:11434/v1/systemone', compactAtPercent: 60, minReductionRatio: 0.25, model: 'kaineone/metzger:0.8b', backend: 'auto' });
    expect(
      resolveHookConfig({ keepThreshold: 0.3, maxStateTokens: 1000, model: 'custom-model', goal: 'g', compactAtPercent: 'no', baseUrl: 'http://localhost:11434/v1/systemone' }),
    ).toEqual({
      baseUrl: 'http://localhost:11434/v1/systemone',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'custom-model',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      backend: 'auto',
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[metzger truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[metzger truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), model: 'kaineone/metzger:0.8b' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('kaineone/metzger:0.8b');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(
      compactSession(transcript(), config, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });

  it('works with localhost, 127.0.0.1 and [::1] endpoints', async () => {
    for (const host of ['http://localhost:11434/v1/systemone', 'http://127.0.0.1:11434/v1/systemone', 'http://[::1]:11434/v1/systemone']) {
      const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), baseUrl: host };
      const { result } = await compactSession(transcript(), config, jevFetch(() => 0.1));
      expect(result.stats.calls).toBeGreaterThan(0);
    }
  });

  it('uses ChatJevAsker for distill models in auto mode', async () => {
    const urls: string[] = [];
    const chatFetch = async (url: string, init?: { body?: string }) => {
      urls.push(url);
      const body = JSON.parse(init?.body ?? '{}');
      const payload = JSON.parse(body.messages[1].content);
      const keepOpt = payload.options.find((o: { key: string }) => o.key === 'true');
      return {
        status: 200,
        ok: true,
        text: JSON.stringify({ message: { content: keepOpt.label } }),
      };
    };
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1 }),
      model: 'tev1-distill:0.8b',
      baseUrl: 'http://localhost:11434/v1/systemone',
    };
    const { result } = await compactSession(transcript(), config, chatFetch);
    expect(urls.every(u => u === 'http://localhost:11434/api/chat')).toBe(true);
    expect(result.decisions.filter(d => d.action === 'keep')).toHaveLength(result.stats.calls);
  });
});

describe('ChatJevAsker', () => {
  it('sends one chat request per noul question and maps A/B to probabilities', async () => {
    const bodies: string[] = [];
    const asker = new ChatJevAsker({
      baseUrl: 'http://localhost:11434',
      model: 'tev1-distill:0.8b',
      fetch: async (_url, init) => {
        bodies.push(init?.body ?? '');
        const body = JSON.parse(init?.body ?? '{}');
        const payload = JSON.parse(body.messages[1].content);
        const dropOpt = payload.options.find((o: { key: string }) => o.key === 'false');
        return {
          status: 200,
          ok: true,
          text: JSON.stringify({ message: { content: dropOpt.label } }),
        };
      },
    });

    const response = await asker.ask('test state', {
      keep_call: { type: 'noul', instructions: 'Should this call be kept?' },
      keep_result: { type: 'noul', instructions: 'Should this result be kept?' },
    });

    expect(bodies).toHaveLength(2);
    expect(response.answers.keep_call).toEqual({ type: 'noul', noul: 0.0 });
    expect(response.answers.keep_result).toEqual({ type: 'noul', noul: 0.0 });
  });

  it('returns 1.0 when the model picks the true/keep option', async () => {
    const asker = new ChatJevAsker({
      baseUrl: 'http://localhost:11434',
      model: 'tev1-distill:0.8b',
      fetch: async (_url, init) => {
        const body = JSON.parse(init?.body ?? '{}');
        const payload = JSON.parse(body.messages[1].content);
        const keepOpt = payload.options.find((o: { key: string }) => o.key === 'true');
        return {
          status: 200,
          ok: true,
          text: JSON.stringify({ message: { content: keepOpt.label } }),
        };
      },
    });

    const response = await asker.ask('test', {
      q: { type: 'noul', instructions: 'Keep?' },
    });

    expect(response.answers.q).toEqual({ type: 'noul', noul: 1.0 });
  });
});

type AnyHook = (...args: any[]) => Promise<any>;

type FakeResponse = { status: number; ok: boolean; text: string };

function setup(
  usagePercent = 90,
  compactImpl: (call: number) => Promise<unknown> = async () => undefined,
  options: Record<string, string | number> = {},
  fetchImpl: (url: string, init?: { method?: string; body?: string }) => Promise<FakeResponse> = async () => {
    throw new Error('no network');
  },
) {
  const hooks = new Map<string, AnyHook>();
  const on = ((pattern: string, hook: AnyHook) => {
    hooks.set(pattern, hook);
    return { dispose() {} };
  }) as never;
  register(on, options);
  const order: string[] = [];
  const timers: { ms: number; fn: () => void }[] = [];
  let compactCalls = 0;
  const $ = {
    session: {
      usage: vi.fn(async () => ({ context: { percent: usagePercent } })),
      compact: vi.fn(async () => {
        order.push('compact');
        return compactImpl(compactCalls++);
      }),
    },
    clock: {
      after: vi.fn((ms: number, fn: () => void) => {
        timers.push({ ms, fn });
        return { cancel() {} };
      }),
    },
    ui: { log: vi.fn(), toast: vi.fn() },
    http: { fetch: vi.fn(fetchImpl) },
    command: { register: vi.fn(async (spec: { name: string }) => ({ command: spec.name })) },
  };
  /** Fires the recorded timers (and any they schedule) in order, letting async work settle. */
  const flush = async (): Promise<void> => {
    while (timers.length > 0) {
      timers.shift()!.fn();
      for (let i = 0; i < 10; i++) await Promise.resolve();
    }
  };
  return { hooks, $, order, timers, flush };
}

describe('turn.complete auto-compact', () => {
  it('skips subagent turns entirely', async () => {
    const { hooks, $, timers, flush } = setup();
    const next = vi.fn(async () => 'r');
    const out = await hooks.get('turn.complete')!($, { agentId: 'sub-1' }, next);
    await flush();
    expect(out).toBe('r');
    expect(next).toHaveBeenCalledTimes(1);
    expect(timers).toHaveLength(0);
    expect($.clock.after).not.toHaveBeenCalled();
    expect($.session.usage).not.toHaveBeenCalled();
    expect($.session.compact).not.toHaveBeenCalled();
  });

  it('calls next first and compacts asynchronously after the hook returns', async () => {
    const { hooks, $, order, flush } = setup();
    const next = vi.fn(async () => {
      order.push('next');
      return 'r';
    });
    const out = await hooks.get('turn.complete')!($, {}, next);
    expect(out).toBe('r');
    expect($.clock.after).toHaveBeenCalledWith(250, expect.any(Function));
    expect($.session.compact).not.toHaveBeenCalled();
    await flush();
    expect($.session.compact).toHaveBeenCalledTimes(1);
    expect(order).toEqual(['next', 'compact']);
  });

  it('does not double-schedule across two turn.complete events', async () => {
    const { hooks, $, flush } = setup();
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    expect($.clock.after).toHaveBeenCalledTimes(1);
    await flush();
    expect($.session.compact).toHaveBeenCalledTimes(1);
    // after it finishes a later turn may schedule again
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    expect($.clock.after).toHaveBeenCalledTimes(2);
  });

  it('retries quietly while the turn is still running, then succeeds', async () => {
    const { hooks, $, timers, flush } = setup(90, async (n) => {
      if (n === 0) throw new Error('metzger: $.session.compact: a turn is running (t1)');
    });
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    await flush();
    expect($.session.compact).toHaveBeenCalledTimes(2);
    expect($.clock.after).toHaveBeenLastCalledWith(1000, expect.any(Function));
    expect($.ui.log).not.toHaveBeenCalled();
    expect(timers).toHaveLength(0);
  });

  it('gives up silently after 3 retries', async () => {
    const { hooks, $, flush } = setup(90, async () => {
      throw new Error('a turn is in flight');
    });
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    await flush();
    expect($.session.compact).toHaveBeenCalledTimes(4);
    expect($.ui.log).not.toHaveBeenCalled();
    // and can schedule again on a later turn
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    expect($.clock.after).toHaveBeenCalledTimes(5);
  });

  it('logs other errors without retrying', async () => {
    const { hooks, $, flush } = setup(90, async () => {
      throw new Error('boom');
    });
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    await flush();
    expect($.session.compact).toHaveBeenCalledTimes(1);
    expect($.ui.log).toHaveBeenCalledWith('auto-compact skipped (boom)');
  });

  it('does not compact below the threshold', async () => {
    const { hooks, $, flush } = setup(10);
    await hooks.get('turn.complete')!($, {}, async () => 'r');
    await flush();
    expect($.session.compact).not.toHaveBeenCalled();
  });
});

describe('session.compact', () => {
  it('passes subagent and fork compactions through untouched', async () => {
    const { hooks, $ } = setup();
    const next = vi.fn(async () => 'core');
    const event = { agentId: 'sub-1', messages: transcript() };
    const out = await hooks.get('session.compact')!($, event, next);
    expect(out).toBe('core');
    expect(next).toHaveBeenCalledWith(event);
    expect($.http.fetch).not.toHaveBeenCalled();
    expect($.ui.log).not.toHaveBeenCalled();
  });
});

describe('systemone request size cap', () => {
  it('throws a clear error instead of sending an oversized body', async () => {
    const fetchFn = vi.fn(async () => ({ status: 200, ok: true, text: '{}' }));
    const asker = jevAsker(fetchFn, 'local', 'kaineone/metzger:0.8b');
    const huge = 'x'.repeat(SYSTEMONE_MAX_BODY_BYTES + 10);
    await expect(
      asker.ask(huge, { q: { type: 'noul', instructions: 'Keep?' } }),
    ).rejects.toThrow(/request too large for local endpoint \(\d+ bytes > \d+ limit\)/);
    expect(fetchFn).not.toHaveBeenCalled();
  });
});

describe('systemone budgets', () => {
  it('keep every request body under the endpoint limit on a large non-ASCII transcript', async () => {
    const messages: SessionMessage[] = [message('user', 'Refactor the é漢字 module.', { handle: 'h-0' })];
    for (let i = 0; i < 150; i++) {
      const blob = { rows: Array.from({ length: 40 }, (_, k) => ({ k, name: `é漢字-${i}-${k}`, ok: k % 2 === 0 })) };
      const text = `${JSON.stringify(blob)} ${'é漢字'.repeat(600)}`;
      messages.push(call(`c${i}`, 'Bash', { command: `echo ${'é漢字'.repeat(40)}`, payload: blob }, text));
      messages.push(result(`c${i}`, text));
      messages.push(message('assistant', `step ${i} ${'é漢字'.repeat(100)}`, { handle: `a-${i}` }));
    }
    const sizes: number[] = [];
    const config = resolveHookConfig({
      preserveRecentMessages: 4,
      maxStateTokens: SYSTEMONE_MAX_STATE_TOKENS,
      maxRequestTokens: SYSTEMONE_MAX_REQUEST_TOKENS,
    });
    await compactSession(messages, config, async (_url, init) => {
      sizes.push(new TextEncoder().encode(init?.body ?? '').length);
      const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
      const answers = Object.fromEntries(
        Object.keys(questions).map((key) => [key, { type: 'noul', noul: 0.5 }]),
      );
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    });
    console.log(`systemone max body bytes: ${Math.max(...sizes)} over ${sizes.length} request(s)`);
    expect(sizes.length).toBeGreaterThan(0);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(SYSTEMONE_MAX_BODY_BYTES);
  });
});

const ok = (text: string): FakeResponse => ({ status: 200, ok: true, text });

/** An Ollama that serves `models`, with `probe` answering the systemone POST. */
function ollama(models: string[], probe: FakeResponse | (() => FakeResponse)) {
  return async (url: string, init?: { method?: string }): Promise<FakeResponse> => {
    if (url.endsWith('/api/version')) return ok('{"version":"0.9.1"}');
    if (url.endsWith('/api/tags')) return ok(JSON.stringify({ models: models.map((name) => ({ name })) }));
    if (url.endsWith('/api/show')) return ok(JSON.stringify({ capabilities: ['decision'], parameters: 'num_ctx 32768' }));
    if (init?.method === 'POST') return typeof probe === 'function' ? probe() : probe;
    throw new Error(`unexpected ${url}`);
  };
}

const runStatus = async (h: ReturnType<typeof setup>): Promise<string> => {
  await h.hooks.get('session.start')!(h.$, {}, async () => 'r');
  const out = await h.hooks.get('command.run')!(h.$, { command: 'status', args: '' }, async () => ({ text: 'builtin' }));
  return out.text as string;
};

describe('status command', () => {
  it('is registered even when baseUrl is not local', async () => {
    const h = setup(90, undefined, { baseUrl: 'https://example.com/v1/systemone' });
    await h.hooks.get('session.start')!(h.$, {}, async () => 'r');
    expect(h.$.command.register).toHaveBeenCalledWith(expect.objectContaining({ name: 'status' }));
  });

  it('falls back to metzger-status when the host refuses the name, and ignores other commands', async () => {
    const h = setup(90, undefined, { baseUrl: 'https://example.com/v1/systemone' });
    h.$.command.register.mockImplementationOnce(async () => {
      throw new Error('built-in');
    });
    await h.hooks.get('session.start')!(h.$, {}, async () => 'r');
    expect(h.$.command.register).toHaveBeenLastCalledWith(expect.objectContaining({ name: 'metzger-status' }));
    const next = vi.fn(async () => ({ text: 'builtin' }));
    await h.hooks.get('command.run')!(h.$, { command: 'status', args: '' }, next);
    expect(next).toHaveBeenCalledTimes(1);
  });

  it('reports a reachability failure when fetch rejects', async () => {
    const h = setup(90, undefined, {}, async () => {
      throw new Error('ECONNREFUSED');
    });
    const text = await runStatus(h);
    expect(text).toContain('ollama: not reachable: ECONNREFUSED');
    expect(text).toContain('last compaction: none yet this session');
    expect(text).toContain('budgets: state 15000');
  });

  it('reports an empty 200 probe body explicitly', async () => {
    const h = setup(90, undefined, {}, ollama(['kaineone/metzger:0.8b'], ok('')));
    const text = await runStatus(h);
    expect(text).toContain('probe: HTTP 200');
    expect(text).toContain('empty response');
    expect(text).toContain('capabilities: decision');
    expect(text).toContain('version 0.9.1');
    expect(text).toContain('budgets: state 15000 tokens, request 18000 tokens');
  });

  it('reports the noul value and flags an invalid answer', async () => {
    const good = setup(90, undefined, {}, ollama(['kaineone/metzger:0.8b'], ok('{"answers":{"q1":{"type":"noul","noul":0.25}}}')));
    expect(await runStatus(good)).toContain('noul=0.25');
    const bad = setup(90, undefined, {}, ollama(['kaineone/metzger:0.8b'], ok('{"answers":{"q1":{"type":"noul","noul":null}}}')));
    expect(await runStatus(bad)).toContain('invalid answer');
  });
});

describe('model names', () => {
  it('uses kaineone/metzger:0.8b when only that is pulled and the legacy name is configured', async () => {
    const h = setup(90, undefined, { model: 'metzger:0.8b' }, ollama(['kaineone/metzger:0.8b'], ok('{}')));
    await h.hooks.get('session.start')!(h.$, {}, async () => 'r');
    expect(h.$.ui.toast).toHaveBeenCalledTimes(1);
    expect(h.$.ui.toast.mock.calls[0]![0]).toBe('metzger: using kaineone/metzger:0.8b; update the model option');
  });

  it('prints the pull command with a single prefix when the model is missing', async () => {
    const h = setup(90, undefined, {}, ollama([], ok('{}')));
    await h.hooks.get('session.start')!(h.$, {}, async () => 'r');
    const text = h.$.ui.toast.mock.calls[0]![0] as string;
    expect(text).toContain('ollama pull kaineone/metzger:0.8b');
    expect(text).not.toContain('kaineone/kaineone');
  });
});

describe('session.compact failures', () => {
  it('surfaces a network error with the url instead of failing silently', async () => {
    const h = setup(90, undefined, {}, async () => {
      throw new Error('ECONNRESET');
    });
    const next = vi.fn(async () => 'core');
    const out = await h.hooks.get('session.compact')!(h.$, { messages: transcript() }, next);
    expect(out).toBe('core');
    const text = h.$.ui.toast.mock.calls[0]![0] as string;
    expect(text).toMatch(/fallback to built-in summary \(request to http:\/\/localhost:11434\/v1\/systemone failed: ECONNRESET\)/);
  });
});
