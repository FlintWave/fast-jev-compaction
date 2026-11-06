import { describe, expect, it, vi } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  jevAsker,
  register,
  SYSTEMONE_MAX_BODY_BYTES,
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

function setup(usagePercent = 90, compactImpl: () => Promise<unknown> = async () => undefined) {
  const hooks = new Map<string, AnyHook>();
  const on = ((pattern: string, hook: AnyHook) => {
    hooks.set(pattern, hook);
    return { dispose() {} };
  }) as never;
  register(on, {});
  const order: string[] = [];
  const $ = {
    session: {
      usage: vi.fn(async () => ({ context: { percent: usagePercent } })),
      compact: vi.fn(async () => {
        order.push('compact');
        return compactImpl();
      }),
    },
    ui: { log: vi.fn(), toast: vi.fn() },
    http: { fetch: vi.fn() },
  };
  return { hooks, $, order };
}

describe('turn.complete auto-compact', () => {
  it('skips subagent turns entirely', async () => {
    vi.useFakeTimers();
    try {
      const { hooks, $ } = setup();
      const next = vi.fn(async () => 'r');
      const out = await hooks.get('turn.complete')!($, { agentId: 'sub-1' }, next);
      await vi.runAllTimersAsync();
      expect(out).toBe('r');
      expect(next).toHaveBeenCalledTimes(1);
      expect($.session.usage).not.toHaveBeenCalled();
      expect($.session.compact).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('calls next first and compacts asynchronously after the hook returns', async () => {
    vi.useFakeTimers();
    try {
      const { hooks, $, order } = setup();
      const next = vi.fn(async () => {
        order.push('next');
        return 'r';
      });
      const out = await hooks.get('turn.complete')!($, {}, next);
      expect(out).toBe('r');
      expect($.session.compact).not.toHaveBeenCalled();
      await vi.runAllTimersAsync();
      expect($.session.compact).toHaveBeenCalledTimes(1);
      expect(order).toEqual(['next', 'compact']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('stays silent when the turn is still running, logs other errors', async () => {
    vi.useFakeTimers();
    try {
      const busy = setup(90, async () => {
        throw new Error('metzger: $.session.compact: a turn is in flight');
      });
      await busy.hooks.get('turn.complete')!(busy.$, {}, async () => 'r');
      await vi.runAllTimersAsync();
      expect(busy.$.ui.log).not.toHaveBeenCalled();

      const bad = setup(90, async () => {
        throw new Error('boom');
      });
      await bad.hooks.get('turn.complete')!(bad.$, {}, async () => 'r');
      await vi.runAllTimersAsync();
      expect(bad.$.ui.log).toHaveBeenCalledWith('auto-compact skipped (boom)');
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not compact below the threshold', async () => {
    vi.useFakeTimers();
    try {
      const { hooks, $ } = setup(10);
      await hooks.get('turn.complete')!($, {}, async () => 'r');
      await vi.runAllTimersAsync();
      expect($.session.compact).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
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
