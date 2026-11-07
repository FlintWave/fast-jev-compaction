import { ChatJevAsker } from './chat-asker.js';
import { buildJevRequest, parseJevResponse } from './request.js';
import type { JevAsker } from './types.js';

export type Backend = 'systemone' | 'chat';
export type BackendOption = Backend | 'auto';

export const DEFAULT_MODEL = 'kaineone/metzger:0.8b';
export const DEFAULT_SYSTEMONE_URL = 'http://localhost:11434/v1/systemone';

/** The minimal fetch the askers need: `$.http.fetch` or a wrapper over `globalThis.fetch`. */
export type FetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};
export type FetchResponse = { status: number; ok: boolean; text: string };
export type FetchLike = (url: string, init?: FetchInit) => Promise<FetchResponse>;

/** The distilled models answer over Ollama's chat API; everything else speaks the decision protocol. */
export function isChatModel(model: string): boolean {
  return model.includes('distill');
}

export function resolveBackend(model: string, backend: BackendOption = 'auto'): Backend {
  if (backend !== 'auto') return backend;
  return isChatModel(model) ? 'chat' : 'systemone';
}

/** The Ollama root (`http://localhost:11434`) for a systemone endpoint URL, or the URL as given. */
export function ollamaBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/v1\/systemone\/?$/, '').replace(/\/+$/, '');
}

/** The local `/v1/systemone` endpoint rejects bodies (text + schema) over 64 KiB; stay under with margin. */
export const SYSTEMONE_MAX_BODY_BYTES = 60 * 1024;
/** Conservative token budgets for the systemone backend so a request fits in SYSTEMONE_MAX_BODY_BYTES. */
export const SYSTEMONE_MAX_REQUEST_TOKENS = 18_000;
export const SYSTEMONE_MAX_STATE_TOKENS = 15_000;

/** A `JevAsker` over the decision-protocol endpoint. */
export function systemOneAsker(
  fetchFn: FetchLike,
  apiKey: string,
  model: string,
  baseUrl?: string,
): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model, baseUrl }, state, questions);
      const bytes = new TextEncoder().encode(request.body).length;
      if (bytes > SYSTEMONE_MAX_BODY_BYTES) {
        throw new Error(
          `request too large for local endpoint (${bytes} bytes > ${SYSTEMONE_MAX_BODY_BYTES} limit)`,
        );
      }
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text, request.url);
    },
  };
}

export type AskerConfig = {
  model?: string;
  /** The systemone endpoint URL; the chat backend derives the Ollama root from it. */
  baseUrl?: string;
  backend?: BackendOption;
  /** Chat backend only: how many questions are in flight at once. */
  concurrency?: number;
};

/** The asker for a model: the chat API for distilled models, the decision protocol otherwise. */
export function createJevAsker(config: AskerConfig, fetchFn: FetchLike): JevAsker {
  const model = config.model ?? DEFAULT_MODEL;
  const baseUrl = config.baseUrl ?? DEFAULT_SYSTEMONE_URL;
  if (resolveBackend(model, config.backend) === 'chat') {
    const options = { baseUrl: ollamaBaseUrl(baseUrl), model, fetch: fetchFn };
    return new ChatJevAsker(
      config.concurrency === undefined ? options : { ...options, concurrency: config.concurrency },
    );
  }
  return systemOneAsker(fetchFn, 'local', model, baseUrl);
}
