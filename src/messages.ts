import { createJevAsker, type AskerConfig } from './backend.js';
import { compact } from './compact.js';
import type { CompactOptions, CompactResult, Message } from './types.js';

export type CompactMessagesOptions = CompactOptions & AskerConfig;

/**
 * `compact` against a local Ollama instance over `globalThis.fetch`. The
 * backend follows the model name: the decision protocol at `baseUrl` for the
 * Metzger models, Ollama's chat API for the distilled ones (`backend` forces
 * either).
 */
export async function compactMessages(
  messages: readonly Message[],
  options: CompactMessagesOptions = {},
): Promise<CompactResult> {
  const asker = createJevAsker(options, async (url, init) => {
    const resp = await globalThis.fetch(url, init);
    return { status: resp.status, ok: resp.ok, text: await resp.text() };
  });
  return compact(messages, asker, options);
}
