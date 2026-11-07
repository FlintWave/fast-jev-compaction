import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import {
  createJevAsker,
  DEFAULT_MODEL,
  DEFAULT_SYSTEMONE_URL,
  ollamaBaseUrl,
  resolveBackend,
  SYSTEMONE_MAX_BODY_BYTES,
  SYSTEMONE_MAX_REQUEST_TOKENS,
  SYSTEMONE_MAX_STATE_TOKENS,
  systemOneAsker,
  type Backend,
  type BackendOption,
} from '../src/backend.js';
import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import type {
  CompactOptions,
  CompactResult,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  baseUrl: DEFAULT_SYSTEMONE_URL,
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  baseUrl: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  backend: BackendOption;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const backendRaw = optionString(options, 'backend');
  const backend = (backendRaw === 'systemone' || backendRaw === 'chat') ? backendRaw : 'auto';
  const config: HookConfig = {
    ...numbers,
    baseUrl: optionString(options, 'baseUrl') ?? HOOK_DEFAULTS.baseUrl,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    backend,
  };
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

export { SYSTEMONE_MAX_BODY_BYTES, SYSTEMONE_MAX_REQUEST_TOKENS, SYSTEMONE_MAX_STATE_TOKENS };

/** A `JevAsker` over the engine's `$.http.fetch`, speaking the decision protocol. */
export const jevAsker = systemOneAsker;

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

function backendOf(config: HookConfig): Backend {
  return resolveBackend(config.model, config.backend);
}

/** Runs the library over a session transcript; throws when Jev fails or the history cannot be fitted. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  const asker = createJevAsker(config, fetchFn);
  const result = await compact(messages, asker, config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} calls dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

/** Keep in sync with .claude-plugin/plugin.json. */
export const METZGER_VERSION = '0.6.1';

const NETWORK_TIMEOUT_MS = 10_000;
const COMPACT_REQUEST_TIMEOUT_MS = 120_000;
const PROBE_TIMEOUT_MS = 60_000;

type Scheduler = (ms: number, fn: () => void) => { cancel: () => void };

/** Rejects when `promise` has not settled after `ms`, using the host clock. */
export function withTimeout<T>(schedule: Scheduler, ms: number, label: string, promise: Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = schedule(ms, () => reject(new Error(`${label} timed out after ${ms} ms`)));
    promise.then(
      (value) => {
        timer.cancel();
        resolve(value);
      },
      (error: unknown) => {
        timer.cancel();
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** The pulled-model name that stands in for `model`, or undefined. Exact name or `:latest` only. */
function findPulled(names: readonly string[], model: string): string | undefined {
  return names.find((n) => n === model || n === `${model}:latest`);
}

/** The one legacy rename: `metzger:0.8b` <-> `kaineone/metzger:0.8b`. */
export function legacyAlias(model: string): string {
  return model.startsWith('kaineone/') ? model.slice('kaineone/'.length) : `kaineone/${model}`;
}

function parseNumCtx(showResp: HookFetchResponse): number {
  if (!showResp.ok) return 0;
  try {
    const data = JSON.parse(showResp.text) as {
      model_info?: Record<string, unknown>;
      parameters?: string;
    };
    if (typeof data.parameters === 'string') {
      const match = data.parameters.match(/^num_ctx\s+(\d+)/m);
      if (match) return parseInt(match[1]!, 10);
    }
    if (data.model_info) {
      for (const [key, value] of Object.entries(data.model_info)) {
        if (key.endsWith('.context_length') && typeof value === 'number') return value;
      }
    }
  } catch {}
  return 0;
}

function parseCapabilities(showResp: HookFetchResponse): string[] | undefined {
  if (!showResp.ok) return undefined;
  try {
    const data = JSON.parse(showResp.text) as { capabilities?: unknown };
    if (Array.isArray(data.capabilities)) return data.capabilities.map(String);
  } catch {}
  return undefined;
}

export type OllamaModelCheck = {
  ollamaUp: boolean;
  modelPulled: boolean;
  numCtx: number;
  /** The pulled name to use: `model`, or its legacy alias when only that is pulled. */
  model: string;
  renamed: boolean;
  capabilities?: string[];
};

async function checkOllamaModel(
  fetchFn: HookFetch,
  baseUrl: string,
  model: string,
): Promise<OllamaModelCheck> {
  const ollamaBase = ollamaBaseUrl(baseUrl);
  const down: OllamaModelCheck = { ollamaUp: false, modelPulled: false, numCtx: 0, model, renamed: false };
  try {
    const resp = await fetchFn(`${ollamaBase}/api/tags`);
    if (!resp.ok) return down;
    const data = JSON.parse(resp.text) as { models?: { name?: string }[] };
    const names = (data.models ?? []).map((m) => m.name ?? '');
    let resolved = findPulled(names, model);
    let renamed = false;
    if (!resolved) {
      resolved = findPulled(names, legacyAlias(model));
      renamed = resolved !== undefined;
    }
    if (!resolved) return { ...down, ollamaUp: true };
    const showResp = await fetchFn(`${ollamaBase}/api/show`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: resolved, name: resolved }),
    });
    const capabilities = parseCapabilities(showResp);
    return {
      ollamaUp: true,
      modelPulled: true,
      numCtx: parseNumCtx(showResp),
      model: renamed ? legacyAlias(model) : model,
      renamed,
      ...(capabilities ? { capabilities } : {}),
    };
  } catch {
    return down;
  }
}

/** The token budgets the next compaction runs with: the backend clamp, then the model's context. */
export function effectiveConfig(configured: HookConfig, modelNumCtx: number): HookConfig {
  const config = { ...configured };
  if (backendOf(config) === 'systemone') {
    config.maxRequestTokens = Math.min(config.maxRequestTokens ?? 30_000, SYSTEMONE_MAX_REQUEST_TOKENS);
    config.maxStateTokens = Math.min(config.maxStateTokens ?? 25_000, SYSTEMONE_MAX_STATE_TOKENS);
  }
  if (modelNumCtx > 0) {
    const maxReq = Math.floor(modelNumCtx * 0.9);
    config.maxRequestTokens = Math.min(config.maxRequestTokens ?? 30_000, maxReq);
    config.maxStateTokens = Math.min(config.maxStateTokens ?? 25_000, maxReq - 500);
  }
  return config;
}

/** One live Jev call, reported as a single line. Never throws. */
export async function probeLine(
  fetchFn: HookFetch,
  config: HookConfig,
): Promise<string> {
  const backend = backendOf(config);
  const started = Date.now();
  const elapsed = (): string => `${Date.now() - started} ms`;
  try {
    if (backend === 'chat') {
      const url = `${ollamaBaseUrl(config.baseUrl)}/api/chat`;
      const response = await fetchFn(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: config.model,
          messages: [{ role: 'user', content: 'Reply with the letter A.' }],
          stream: false,
          options: { temperature: 0 },
          think: false,
        }),
      });
      if (!response.ok) return `chat probe: HTTP ${response.status} in ${elapsed()}: ${response.text.slice(0, 200)}`;
      if (response.text.trim() === '') return `chat probe: empty response from ${url} (HTTP ${response.status}, ${elapsed()})`;
      let content: unknown;
      try {
        content = (JSON.parse(response.text) as { message?: { content?: unknown } }).message?.content;
      } catch {
        return `chat probe: malformed JSON in ${elapsed()}: ${response.text.slice(0, 200)}`;
      }
      if (typeof content !== 'string' || content.trim() === '') {
        return `chat probe: empty response from ${url} (HTTP ${response.status}, ${elapsed()})`;
      }
      return `chat probe: HTTP ${response.status} in ${elapsed()}, model answered "${content.trim().slice(0, 40)}"`;
    }
    const response = await fetchFn(config.baseUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer local' },
      body: JSON.stringify({
        model: config.model,
        state: { history: [{ role: 'user', text: 'ping' }] },
        questions: { q1: { type: 'noul', instructions: 'Is this message still needed?' } },
      }),
    });
    const prefix = `probe: HTTP ${response.status} in ${elapsed()}`;
    if (!response.ok) return `${prefix}, error: ${response.text.slice(0, 200)}`;
    if (response.text.trim() === '') return `${prefix}, empty response`;
    let parsed: { answers?: Record<string, { noul?: unknown }> };
    try {
      parsed = JSON.parse(response.text);
    } catch {
      return `${prefix}, malformed JSON: ${response.text.slice(0, 200)}`;
    }
    const noul = parsed.answers?.q1?.noul;
    if (typeof noul !== 'number' || !Number.isFinite(noul)) {
      return `${prefix}, invalid answer: ${response.text.slice(0, 200)}`;
    }
    return `${prefix}, noul=${noul}`;
  } catch (error) {
    return `probe: failed after ${elapsed()}: ${errorMessage(error)}`;
  }
}

const COMPACT_DELAY_MS = 250;
const COMPACT_RETRY_MS = 1000;
const COMPACT_RETRIES = 3;
/** A scheduled auto-compact older than this is taken as lost (its timer never fired) and may be scheduled again. */
const SCHEDULE_STALE_MS = 5 * 60_000;

/** The plugin's file command `commands/status.md`, namespaced by the host as `/metzger:status`. */
const STATUS_COMMAND = 'metzger:status';

/** `$.session.compact` rejects while a turn runs; the exact wording is the host's, so match loosely. */
function isTurnRunningError(message: string): boolean {
  return /turn|in flight|busy|not idle/i.test(message);
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  let scheduledAt: number | undefined;
  let setupChecked = false;
  let modelNumCtx = 0;
  let lastCompaction: { at: number; text: string } | undefined;
  const record = (text: string): void => {
    lastCompaction = { at: Date.now(), text };
  };

  on('session.start', async ($, event, next) => {
    const result = await next(event);

    if (setupChecked) return result;
    setupChecked = true;

    const isLocal = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/.test(configured.baseUrl);
    if (!isLocal) return result;

    // The engine awaits session.start before the first prompt, so the network
    // check runs from a timer and never holds the session up.
    const checkSetup = async (): Promise<void> => {
      const check = await checkOllamaModel(
        (url, init) =>
          withTimeout((ms, fn) => $.clock.after(ms, fn), NETWORK_TIMEOUT_MS, `request to ${url}`, $.http.fetch(url, init)).then((r) => ({
            status: r.status,
            ok: r.ok,
            text: r.text,
          })),
        configured.baseUrl,
        configured.model,
      );
      modelNumCtx = check.numCtx;

      if (!check.ollamaUp) {
        $.ui.toast(
          `metzger: Ollama is not running. Install it from https://ollama.com then run: ollama pull ${configured.model}`,
          { timeoutMs: 30_000 },
        );
        return;
      }

      if (!check.modelPulled) {
        $.ui.toast(
          `metzger: decision model "${configured.model}" not found. Run: ollama pull ${configured.model}`,
          { timeoutMs: 30_000 },
        );
        return;
      }

      if (check.renamed) {
        $.ui.toast(`metzger: using ${check.model}; update the model option`, { timeoutMs: 15_000 });
        configured.model = check.model;
      }
    };
    $.clock.after(0, () => {
      void checkSetup();
    });

    return result;
  });

  on('command.run', { command: STATUS_COMMAND }, async ($, event, next) => {
    if (event.command !== STATUS_COMMAND) return next(event);
    const fetchFn = (url: string, init?: HookFetchInit, ms = NETWORK_TIMEOUT_MS): Promise<HookFetchResponse> =>
      withTimeout((ms, fn) => $.clock.after(ms, fn), ms, `request to ${url}`, $.http.fetch(url, init)).then((r) => ({
        status: r.status,
        ok: r.ok,
        text: r.text,
      }));
    const backend = backendOf(configured);
    const lines = [
      `metzger ${METZGER_VERSION}`,
      `backend: ${backend}`,
      `model: ${configured.model}`,
      `baseUrl: ${configured.baseUrl}`,
    ];

    const base = ollamaBaseUrl(configured.baseUrl);
    let reachable = false;
    try {
      const response = await fetchFn(`${base}/api/version`);
      if (!response.ok) {
        lines.push(`ollama: not reachable: HTTP ${response.status} from ${base}/api/version`);
      } else {
        reachable = true;
        let version = response.text.trim().slice(0, 40);
        try {
          const parsed = JSON.parse(response.text) as { version?: unknown };
          if (typeof parsed.version === 'string') version = parsed.version;
        } catch {}
        lines.push(`ollama: reachable at ${base} (version ${version || 'unknown'})`);
      }
    } catch (error) {
      lines.push(`ollama: not reachable: ${errorMessage(error)}`);
    }

    let numCtx = modelNumCtx;
    if (reachable) {
      const check = await checkOllamaModel(fetchFn, configured.baseUrl, configured.model);
      if (!check.modelPulled) {
        lines.push(`model: not pulled (run: ollama pull ${configured.model})`);
      } else {
        numCtx = check.numCtx || numCtx;
        modelNumCtx = numCtx;
        const note = check.renamed ? ` (pulled as ${check.model}; update the model option)` : '';
        lines.push(`model: pulled${note}, num_ctx ${check.numCtx || 'unknown'}`);
        if (backend === 'systemone') {
          if (!check.capabilities) {
            lines.push('capabilities: not reported by Ollama');
          } else {
            lines.push(`capabilities: ${check.capabilities.join(', ') || '(none)'}`);
            if (!check.capabilities.includes('decision')) {
              lines.push(
                `WARNING: the model lacks the "decision" capability, which /v1/systemone needs. Pull ${DEFAULT_MODEL}.`,
              );
            }
          }
        }
      }
      lines.push(await probeLine((url, init) => fetchFn(url, init, PROBE_TIMEOUT_MS), configured));
    } else {
      lines.push('model and probe: skipped (Ollama not reachable)');
    }

    const effective = effectiveConfig(configured, numCtx);
    lines.push(
      `budgets: state ${effective.maxStateTokens ?? 25_000} tokens, request ${effective.maxRequestTokens ?? 30_000} tokens`,
    );
    let context = 'unknown';
    try {
      const usage = await $.session.usage();
      if (usage.context.percent !== undefined && usage.context.percent !== null) {
        context = `${Math.round(usage.context.percent)}%`;
      }
    } catch {}
    lines.push(`context: ${context} (auto-compact at ${configured.compactAtPercent}%)`);
    lines.push(
      lastCompaction
        ? `last compaction: ${new Date(lastCompaction.at).toISOString()} ${lastCompaction.text}`
        : 'last compaction: none yet this session',
    );
    return { text: lines.join('\n') };
  });

  on('session.compact', async ($, event, next) => {
    // A subagent's or fork's own transcript: leave it to core.
    if (event.agentId) return next(event);
    // A precompute installs nothing and runs ahead of time; Metzger decides at
    // the real compaction, when the transcript is final. Core may still prepare
    // the summary the fallback path would use.
    if (event.trigger === 'precompute') return next(event);
    try {
      const config = effectiveConfig(configured, modelNumCtx);
      // What the person typed after /compact is what to keep in mind while scoring.
      if (!config.goal && event.instructions && event.instructions.trim().length > 0) {
        config.goal = event.instructions.trim();
      }
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        let response;
        try {
          response = await withTimeout(
            (ms, fn) => $.clock.after(ms, fn),
            COMPACT_REQUEST_TIMEOUT_MS,
            `request to ${url}`,
            $.http.fetch(url, init),
          );
        } catch (error) {
          throw new Error(`request to ${url} failed: ${errorMessage(error)}`);
        }
        return { status: response.status, ok: response.ok, text: response.text };
      });
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        const text = `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`;
        record(text);
        notify($, text);
        return next(event);
      }
      const text = `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`;
      record(text);
      notify($, text);
      return { messages };
    } catch (error) {
      const text = `fallback to built-in summary (${errorMessage(error)})`;
      record(text);
      notify($, text);
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    // Subagent turns reach every hook; only the main loop may trigger compaction.
    if (event.agentId) return next(event);
    const result = await next(event);
    const now = Date.now();
    const pending = scheduledAt !== undefined && now - scheduledAt < SCHEDULE_STALE_MS;
    if (compacting || pending) return result;
    scheduledAt = now;

    // $.session.compact rejects while a turn runs, so start it after this hook
    // returns, and retry a few times if the host has not yet finished the turn.
    const run = async (attempt: number): Promise<void> => {
      let retry = false;
      try {
        const { context } = await $.session.usage();
        if ((context.percent ?? 0) < configured.compactAtPercent) return;
        compacting = true;
        const outcome = await $.session.compact();
        if (outcome && 'skip' in outcome && outcome.skip) {
          record(`auto-compact skipped by a hook (${outcome.skip})`);
        }
      } catch (error) {
        const message = errorMessage(error);
        if (isTurnRunningError(message)) {
          retry = attempt < COMPACT_RETRIES;
          if (!retry) record(`auto-compact gave up: a turn was still running after ${COMPACT_RETRIES} retries`);
        } else {
          record(`auto-compact skipped (${message})`);
          $.ui.log(`auto-compact skipped (${message})`);
        }
      } finally {
        compacting = false;
        if (!retry) scheduledAt = undefined;
      }
      if (retry) {
        $.clock.after(COMPACT_RETRY_MS, () => {
          void run(attempt + 1);
        });
      }
    };
    $.clock.after(COMPACT_DELAY_MS, () => {
      void run(0);
    });
    return result;
  });
};

export { resolveOptions };
