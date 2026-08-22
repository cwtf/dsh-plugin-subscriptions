/**
 * dsh-plugin-subscriptions: register OAuth-subscription LLM providers
 * (ChatGPT/Codex, Claude, Grok, Gemini) on `ctx.llm`, and expose the `/subscriptions-auth`
 * RPC channel the web Settings page uses to run the logins. The token store
 * lives at `~/.dsh/plugins/subscriptions/auth.json`; the channel registers only when
 * a host `connection` service exists, so headless compositions load fine.
 * @module dsh-plugin-subscriptions
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { errorChain } from '@deepseek-ai/dsh-llm'
import type { AdapterRegistrationHandle } from '@deepseek-ai/dsh-llm'
// Type-only: activates the `ctx.tools` Context merge for the inject block.
import type {} from '@deepseek-ai/dsh-tools'
import type { AttachmentStore, ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import { OAuthFlowManager, type OAuthAttempt } from './auth/oauth-flow.js'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { readClaudeCodeCredentials, refreshClaudeSynced } from './auth/claude-code-creds.js'
import { registerAuthRpc } from './auth/rpc.js'
import type {
  AuthController,
  ImageBytesResult,
  ProviderStatus,
  SpeedController,
  SpeedTier,
  VideoBytesResult,
} from './auth/rpc.js'
import {
  deleteSession,
  getSession,
  saveSession,
  PROVIDER_IDS,
} from './auth/store.js'
import type {
  ClaudeSession,
  CodexSession,
  GeminiSession,
  GrokSession,
  ProviderId,
  SessionMap,
  StoredSession,
} from './auth/store.js'
import { TokenManager, validateModels } from './providers/common.js'
import type { ModelEntry, ProviderUsage } from './providers/common.js'
import { catalogStore } from './providers/catalog-store.js'
import {
  CodexAdapter,
  codexFlow,
  CODEX_PREEMPT_MS,
  codexProfileClaims,
  exchangeCodexCode,
  fetchCodexUsage,
  isCodexPermanentRefreshError,
  refreshCodex,
} from './providers/codex.js'
import {
  ClaudeAdapter,
  claudeFlow,
  CLAUDE_PREEMPT_MS,
  exchangeClaudeCode,
  fetchClaudeUsage,
  isClaudePermanentRefreshError,
  refreshClaude,
} from './providers/claude.js'
import {
  GrokAdapter,
  grokFlow,
  GROK_PREEMPT_MS,
  exchangeGrokCode,
  fetchGrokUsage,
  isGrokPermanentRefreshError,
  refreshGrok,
} from './providers/grok.js'
import {
  GeminiAdapter,
  geminiFlow,
  GEMINI_PREEMPT_MS,
  exchangeGeminiCode,
  fetchGeminiUsage,
  isGeminiPermanentRefreshError,
  refreshGemini,
} from './providers/gemini.js'
import { createXSearchTool } from './tools/x-search.js'
import { createImageGenerateTool } from './tools/image-generate.js'
import { createVideoGenerateTool, videosDirectory } from './tools/video-generate.js'

export type { ModelEntry, ProviderUsage, UsageWindow } from './providers/common.js'
export type { ProviderStatus } from './auth/rpc.js'
export type { ClaudeSession, CodexSession, GeminiSession, GrokSession, ProviderId } from './auth/store.js'

export const name = 'dsh-plugin-subscriptions'
export const inject = ['llm']

/** Default maximum provider idle time while one stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000

/** Plugin config, validated by the same-named schemastery schema. */
export interface Config {
  /** Provider routes to register; defaults to all four. */
  providers?: ProviderId[]
  /** Maximum provider idle time while one stream read is outstanding (default five minutes). */
  streamIdleTimeoutMs?: number
  /** Advisory model catalogs overriding the built-in defaults, per provider. */
  models?: {
    codex?: ModelEntry[]
    claude?: ModelEntry[]
    grok?: ModelEntry[]
    gemini?: ModelEntry[]
  }
}

const providerIdSchema = z.union(['codex', 'claude', 'grok', 'gemini'])
const modelEntrySchema: z<ModelEntry> = z.object({
  id: z.string().required(),
  name: z.string(),
  contextWindow: z.number().step(1).min(1),
  maxTokens: z.number().step(1).min(1),
  inputModalities: z.array(z.union(['text', 'image'])),
})

export const Config: z<Config> = z.object({
  providers: z.array(providerIdSchema).default(['codex', 'claude', 'grok', 'gemini']),
  streamIdleTimeoutMs: z.number().min(1).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  models: z.object({
    codex: z.array(modelEntrySchema),
    claude: z.array(modelEntrySchema),
    grok: z.array(modelEntrySchema),
    gemini: z.array(modelEntrySchema),
  }),
})

/** Built-in catalogs used when the config does not override a provider's models. */
const DEFAULT_MODELS: Record<ProviderId, ModelEntry[]> = {
  codex: [
    { id: 'gpt-5.1-codex', name: 'GPT-5.1 Codex' },
    { id: 'gpt-5.1-codex-mini', name: 'GPT-5.1 Codex Mini' },
    { id: 'gpt-5.1', name: 'GPT-5.1' },
  ],
  claude: [
    { id: 'claude-opus-5', name: 'Claude Opus 5', maxTokens: 128_000, contextWindow: 1_000_000 },
    { id: 'claude-sonnet-5', name: 'Claude Sonnet 5', maxTokens: 128_000, contextWindow: 1_000_000 },
    { id: 'claude-fable-5', name: 'Claude Fable 5', maxTokens: 128_000, contextWindow: 1_000_000 },
    { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', maxTokens: 64_000 },
  ],
  grok: [
    { id: 'grok-4', name: 'Grok 4' },
    { id: 'grok-4-fast-reasoning', name: 'Grok 4 Fast Reasoning' },
    { id: 'grok-code-fast-1', name: 'Grok Code Fast 1' },
  ],
  gemini: [
    { id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro Preview', maxTokens: 65_536, contextWindow: 1_000_000 },
    { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash Preview', maxTokens: 65_536, contextWindow: 1_000_000 },
    { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', maxTokens: 65_536, contextWindow: 1_000_000 },
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', maxTokens: 65_536, contextWindow: 1_000_000 },
    { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite', maxTokens: 65_536, contextWindow: 1_000_000 },
  ],
}

/** Validate and detach the model catalog for every provider. */
function resolveCatalog(models: Config['models']): Record<ProviderId, ModelEntry[]> {
  const resolve = (provider: ProviderId): ModelEntry[] => {
    // Schemastery injects `[]` for omitted array fields, so an empty list
    // cannot be told apart from an absent one: both mean the built-ins.
    const configured = models?.[provider]
    const entries = configured !== undefined && configured.length > 0 ? configured : DEFAULT_MODELS[provider]
    return validateModels(entries, `${name}: models.${provider}`)
  }
  return {
    codex: resolve('codex'),
    claude: resolve('claude'),
    grok: resolve('grok'),
    gemini: resolve('gemini'),
  }
}

/** The display account of a stored session, for the status endpoint. */
function accountOf(provider: ProviderId, session: StoredSession | undefined): string | undefined {
  if (session === undefined) return undefined
  switch (provider) {
    case 'codex': {
      const codex = session as CodexSession
      // Sessions stored before identity claims were persisted still carry the
      // id token: decode the email on the fly instead of forcing a re-login.
      return codex.emailAddress ?? codexProfileClaims(codex.idToken).emailAddress ?? codex.accountId
    }
    case 'claude': return (session as ClaudeSession).emailAddress
    case 'grok': return (session as GrokSession).account
    case 'gemini': return (session as GeminiSession).emailAddress
  }
}

/** Per-provider usage lookup; providers without a usage endpoint are absent. */
type UsageFetchers = Partial<Record<ProviderId, (signal: AbortSignal) => Promise<ProviderUsage>>>

/**
 * Auth operations behind the `/subscriptions-auth` RPC channel: start/complete
 * OAuth attempts in the background, feed pasted codes, cancel, log out, and
 * answer usage lookups.
 *
 * @internal Exported for tests only; not part of the plugin's public surface.
 */
export class SubscriptionsAuthController implements AuthController {
  /** Last login failure per provider, surfaced as `detail` until the next success. */
  private lastError = new Map<ProviderId, string>()

  /** In-flight OAuth completions, one per provider at most. */
  private completions = new Map<ProviderId, Promise<void>>()

  /**
   * Per-provider claim counter. Everything that takes ownership of a
   * provider's session — starting a login, importing Claude Code credentials,
   * cancelling, logging out — bumps it, and a session write carrying an older
   * number has been superseded and is dropped.
   *
   * The counter is what makes a late OAuth completion safe: an attempt leaves
   * `OAuthFlowManager`'s pending map the moment its callback delivers the
   * code, while the token exchange that follows can still run for seconds. For
   * that whole window `pending(provider)?.cancel()` is a no-op, so ownership
   * cannot be read off the flow manager.
   */
  private claims = new Map<ProviderId, number>()

  /**
   * Token exchanges currently running, per provider. An attempt leaves the
   * flow manager the moment its callback delivers the code, so `flows.isBusy`
   * goes false while the exchange that follows can still run for seconds —
   * gemini's first login provisions a Code Assist project and takes the
   * longest. Counted rather than flagged so a superseded exchange finishing
   * cannot clear the flag on the one that replaced it.
   */
  private exchanges = new Map<ProviderId, number>()

  constructor(
    private readonly flows: OAuthFlowManager,
    /** Announces a provider's auth-state change so catalog readers re-query (fires `llm/adapters-updated`). */
    private readonly onAuthChanged: (provider: ProviderId) => void,
    /** Lazy attachment-store lookup for the `image` endpoint. */
    private readonly resolveAttachments: () => AttachmentStore | undefined,
    /** Usage lookups for providers that expose a usage endpoint. */
    private readonly usageFetchers: UsageFetchers = {},
    /**
     * Reads the Claude Code session from its own store. Constructor-injected so
     * tests can drive both login paths without a real credential store; the
     * plugin itself always uses the default.
     */
    private readonly readClaudeCreds: () => ClaudeSession | undefined = readClaudeCodeCredentials,
  ) {}

  usage(provider: ProviderId, signal: AbortSignal): Promise<ProviderUsage> {
    const fetcher = this.usageFetchers[provider]
    if (fetcher === undefined) return Promise.resolve({ supported: false })
    return fetcher(signal)
  }

  async readImage(ref: ImageAttachmentRef, signal: AbortSignal): Promise<ImageBytesResult> {
    const attachments = this.resolveAttachments()
    if (attachments === undefined) {
      throw new Error('no attachment service is mounted; generated-image bytes are unavailable')
    }
    const stored = await attachments.readImage(ref, signal)
    return { mediaType: stored.ref.mediaType, dataBase64: Buffer.from(stored.data).toString('base64') }
  }

  async readVideo(name: string, signal: AbortSignal): Promise<VideoBytesResult> {
    // The RPC layer validated `name` down to a bare file name, so this join
    // cannot escape the videos directory.
    const data = await readFile(join(videosDirectory(), name), { signal })
    return { mediaType: 'video/mp4', dataBase64: data.toString('base64') }
  }

  async status(provider: ProviderId): Promise<ProviderStatus> {
    const session = await getSession(provider)
    const account = accountOf(provider, session)
    // The plan name is shown by the usage section, so `detail` only carries errors.
    const detail = this.lastError.get(provider)
    return {
      loggedIn: session !== undefined,
      busy: this.flows.isBusy(provider) || (this.exchanges.get(provider) ?? 0) > 0,
      ...session === undefined ? {} : { expiresAt: session.expiresAt },
      ...account === undefined ? {} : { account },
      ...detail === undefined ? {} : { detail },
    }
  }

  async login(provider: ProviderId): Promise<{ authorizeUrl: string }> {
    if (provider === 'claude') {
      const imported = this.readClaudeCreds()
      if (imported !== undefined) {
        // An OAuth attempt may be in flight from an earlier click — the user
        // logged in through the CLI meanwhile. Claiming supersedes it whether
        // it is still waiting for its code or already exchanging one; the
        // cancel on top of that frees the listener, so `busy` clears and the
        // still-open browser tab cannot finish the flow.
        this.claim('claude')
        this.flows.pending('claude')?.cancel()
        await this.persist('claude', imported)
        this.lastError.delete('claude')
        this.onAuthChanged('claude')
        return { authorizeUrl: '' }
      }
      // No Claude Code CLI / credential store — fall back to interactive OAuth.
      const attempt = await this.flows.start('claude', claudeFlow)
      this.completions.set('claude', this.complete('claude', attempt, this.claim('claude')))
      return { authorizeUrl: attempt.authorizeUrl }
    }
    const spec = provider === 'grok' ? await grokFlow() : provider === 'gemini' ? geminiFlow : codexFlow
    const attempt = await this.flows.start(provider, spec)
    // Claimed only once the attempt exists: a rejected `start()` (one attempt
    // per provider) must not supersede the attempt already running.
    this.completions.set(provider, this.complete(provider, attempt, this.claim(provider)))
    return { authorizeUrl: attempt.authorizeUrl }
  }

  /**
   * Take ownership of a provider's session, superseding every older claim.
   * @param provider - the provider route.
   * @returns the claim number a later write checks itself against.
   */
  private claim(provider: ProviderId): number {
    const next = (this.claims.get(provider) ?? 0) + 1
    this.claims.set(provider, next)
    return next
  }

  /**
   * Drive one attempt to a stored session; records failures for the status
   * endpoint. The exchange runs unsupervised — the attempt is gone from the
   * flow manager as soon as its code arrives — so the result is stored only
   * while `claim` still owns the provider's session.
   */
  private async complete(provider: ProviderId, attempt: OAuthAttempt, claim: number): Promise<void> {
    let exchanging = false
    try {
      const code = await attempt.waitCode()
      // From here the flow manager no longer reports this provider busy, but
      // the login is still running: keep the card in its pending state.
      this.exchanges.set(provider, (this.exchanges.get(provider) ?? 0) + 1)
      exchanging = true
      const session = await this.exchange(provider, code, attempt)
      // Whoever claimed the session while the exchange ran owns it now, and
      // this result is stale. The check and the store call sit in one
      // synchronous stretch, and the store queues a write the moment it is
      // called, so a claim arriving after the check is ordered after this
      // write too.
      if (this.claims.get(provider) !== claim) return
      await this.persist(provider, session)
      this.lastError.delete(provider)
      this.onAuthChanged(provider)
    } catch (error) {
      // A failure is as stale as a success would have been: whoever claimed
      // the session while the exchange ran owns what the card shows, so a
      // superseded attempt must not put an error on a provider that has since
      // been imported, logged in again, or logged out.
      if (this.claims.get(provider) !== claim) return
      // A user-cancelled attempt is not a failure worth surfacing. Every
      // in-tree canceller claims first, so the guard above already covers
      // this; the check stands on its own so the invariant does not depend on
      // callers ordering the two.
      if (!(error instanceof Error && error.message === 'login cancelled')) {
        this.lastError.set(provider, errorChain(error))
      }
    } finally {
      if (exchanging) {
        const running = (this.exchanges.get(provider) ?? 1) - 1
        if (running > 0) this.exchanges.set(provider, running)
        else this.exchanges.delete(provider)
      }
    }
  }

  private exchange(provider: ProviderId, code: string, attempt: OAuthAttempt): Promise<StoredSession> {
    switch (provider) {
      case 'codex':
        return exchangeCodexCode(code, attempt.pkce.verifier, attempt.redirectUri)
      case 'claude':
        return exchangeClaudeCode(code, attempt.pkce.verifier, attempt.redirectUri, attempt.state)
      case 'grok':
        return exchangeGrokCode(code, attempt.pkce.verifier, attempt.redirectUri, attempt.pkce.challenge)
      case 'gemini':
        return exchangeGeminiCode(code, attempt.pkce.verifier, attempt.redirectUri)
    }
  }

  private persist(provider: ProviderId, session: StoredSession): Promise<void> {
    // The switch keeps the generic key and the session type aligned.
    switch (provider) {
      case 'codex': return saveSession('codex', session as SessionMap['codex'] & object)
      case 'claude': return saveSession('claude', session as SessionMap['claude'] & object)
      case 'grok': return saveSession('grok', session as SessionMap['grok'] & object)
      case 'gemini': return saveSession('gemini', session as SessionMap['gemini'] & object)
    }
  }

  /**
   * Settle once no OAuth completion is running for a provider.
   *
   * @internal Exported for tests only: a login's token exchange outlives the
   * `login()` call that started it, and a test asserting on what it stored
   * would otherwise have to guess at a timeout.
   */
  async settled(provider: ProviderId): Promise<void> {
    await this.completions.get(provider)
  }

  manual(provider: ProviderId, input: string): Promise<void> {
    const attempt = this.flows.pending(provider)
    if (attempt === undefined) {
      return Promise.reject(new Error(`no ${provider} login attempt is in progress`))
    }
    attempt.manual(input)
    return Promise.resolve()
  }

  cancel(provider: ProviderId): Promise<void> {
    // Claiming covers the attempt whose code already arrived: it is no longer
    // pending, but its token exchange may still be on its way to a store write.
    this.claim(provider)
    this.flows.pending(provider)?.cancel()
    return Promise.resolve()
  }

  async logout(provider: ProviderId): Promise<void> {
    this.claim(provider)
    this.flows.pending(provider)?.cancel()
    await deleteSession(provider)
    this.lastError.delete(provider)
    this.onAuthChanged(provider)
  }
}

export function apply(ctx: Context, config: Config): void {
  const providers = [...new Set(config.providers ?? [...PROVIDER_IDS])]
  const streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
  if (!Number.isFinite(streamIdleTimeoutMs) || streamIdleTimeoutMs <= 0) {
    throw new Error(`${name}: streamIdleTimeoutMs must be a positive finite number`)
  }
  const catalog = resolveCatalog(config.models)
  // A non-empty configured catalog is an explicit override: it wins over live
  // discovery entirely (schemastery injects [] for omitted arrays, so only a
  // non-empty list counts as configured).
  const overridden = new Set<ProviderId>(
    PROVIDER_IDS.filter(provider => (config.models?.[provider]?.length ?? 0) > 0),
  )
  const flows = new OAuthFlowManager()
  const onWarn = (message: string): void => {
    ctx.logger.warn(`dsh-plugin-subscriptions: ${message}`)
  }
  // Optional: resolves ImageBlock references to bytes for vision-capable
  // models. Resolved per request — the attachments service may start after
  // this plugin's apply, so a one-time capture would stay undefined forever.
  const resolveAttachments = (): AttachmentStore | undefined =>
    ctx.get('attachments') as AttachmentStore | undefined

  // Registration handles are kept so an auth-state change can re-announce the
  // route (`replace` fires `llm/adapters-updated`), which makes the web model
  // picker re-query `listModels` and show/hide the provider.
  const handles = new Map<ProviderId, AdapterRegistrationHandle>()
  const authChanged = (provider: ProviderId): void => {
    handles.get(provider)?.replace([provider])
  }
  // Token managers double as the tools' credential source, so they are
  // captured beside the registrations for the inject block below.
  let codexTokens: TokenManager<CodexSession> | undefined
  let claudeTokens: TokenManager<ClaudeSession> | undefined
  let grokTokens: TokenManager<GrokSession> | undefined
  // Usage lookups resolve the session through the refresh-aware path, so an
  // expired access token renews instead of failing the lookup.
  const usageFetchers: UsageFetchers = {}
  // The composer Speed toggle's state: per-session, in-memory (a restart
  // restores standard routing), gated per request on the model's discovered
  // fast-tier support so a stale choice cannot leak onto a plain model.
  const speedBySession = new Map<string, SpeedTier>()
  let codexAdapter: CodexAdapter | undefined

  for (const provider of providers) {
    switch (provider) {
      case 'codex': {
        const tokens = new TokenManager<CodexSession>({
          displayName: 'ChatGPT (Codex)',
          preemptMs: CODEX_PREEMPT_MS,
          load: () => getSession('codex'),
          save: session => saveSession('codex', session),
          remove: () => deleteSession('codex'),
          refresh: refreshCodex,
          isPermanent: isCodexPermanentRefreshError,
          onRemoved: () => { authChanged('codex') },
        })
        codexTokens = tokens
        usageFetchers.codex = async signal => fetchCodexUsage(await tokens.session(), fetch, signal)
        let adapter!: CodexAdapter
        adapter = new CodexAdapter({
          models: catalog.codex,
          streamIdleTimeoutMs,
          tokens,
          discovery: !overridden.has('codex'),
          onWarn,
          resolveAttachments,
          // Durable catalog: capability metadata (reasoning efforts) survives
          // restarts, so a resumed session's selected effort keeps resolving.
          catalogStore: catalogStore('codex'),
          speedFor: (sessionId: string | undefined, model: string): boolean | Promise<boolean> =>
            sessionId !== undefined
            && speedBySession.get(sessionId) === 'fast'
            && adapter.supportsFastTier(model),
        })
        codexAdapter = adapter
        handles.set('codex', ctx.llm.registerAdapter(['codex'], adapter))
        break
      }
      case 'claude': {
        const tokens = new TokenManager<ClaudeSession>({
          displayName: 'Claude (Subscription)',
          preemptMs: CLAUDE_PREEMPT_MS,
          load: () => getSession('claude'),
          save: session => saveSession('claude', session),
          remove: () => deleteSession('claude'),
          refresh: session => refreshClaudeSynced(session, refreshClaude),
          isPermanent: isClaudePermanentRefreshError,
          onRemoved: () => { authChanged('claude') },
        })
        claudeTokens = tokens
        usageFetchers.claude = async signal => fetchClaudeUsage(await tokens.session(), fetch, signal)
        handles.set('claude', ctx.llm.registerAdapter(['claude'], new ClaudeAdapter({
          models: catalog.claude,
          streamIdleTimeoutMs,
          tokens,
          discovery: !overridden.has('claude'),
          onWarn,
          maxRetries: 10,
          resolveAttachments,
          catalogStore: catalogStore('claude'),
        })))
        break
      }
      case 'grok': {
        const tokens = new TokenManager<GrokSession>({
          displayName: 'Grok (Subscription)',
          preemptMs: GROK_PREEMPT_MS,
          load: () => getSession('grok'),
          save: session => saveSession('grok', session),
          remove: () => deleteSession('grok'),
          refresh: refreshGrok,
          isPermanent: isGrokPermanentRefreshError,
          onRemoved: () => { authChanged('grok') },
        })
        grokTokens = tokens
        usageFetchers.grok = async signal => fetchGrokUsage(await tokens.session(), fetch, signal)
        handles.set('grok', ctx.llm.registerAdapter(['grok'], new GrokAdapter({
          models: catalog.grok,
          streamIdleTimeoutMs,
          tokens,
          discovery: !overridden.has('grok'),
          onWarn,
          resolveAttachments,
          // Durable catalog: capability metadata (reasoning efforts) survives
          // restarts, so a resumed session's selected effort keeps resolving.
          catalogStore: catalogStore('grok'),
        })))
        break
      }
      case 'gemini': {
        const tokens = new TokenManager<GeminiSession>({
          displayName: 'Gemini (Subscription)',
          preemptMs: GEMINI_PREEMPT_MS,
          load: () => getSession('gemini'),
          save: session => saveSession('gemini', session),
          remove: () => deleteSession('gemini'),
          refresh: refreshGemini,
          isPermanent: isGeminiPermanentRefreshError,
          onRemoved: () => { authChanged('gemini') },
        })
        usageFetchers.gemini = async signal => fetchGeminiUsage(await tokens.session(), fetch, signal)
        // Code Assist has no model-list endpoint: the catalog is the
        // configured/static one, so there is no discovery to gate or persist.
        handles.set('gemini', ctx.llm.registerAdapter(['gemini'], new GeminiAdapter({
          models: catalog.gemini,
          streamIdleTimeoutMs,
          tokens,
          resolveAttachments,
        })))
        break
      }
    }
  }

  const speed: SpeedController = {
    async speed(sessionId) {
      return {
        tier: speedBySession.get(sessionId) ?? 'standard',
        fastModels: await codexAdapter?.fastCapableModels() ?? [],
      }
    },
    async setSpeed(sessionId, tier) {
      if (tier === 'standard') speedBySession.delete(sessionId)
      else speedBySession.set(sessionId, tier)
    },
  }
  registerAuthRpc(ctx, new SubscriptionsAuthController(flows, authChanged, resolveAttachments, usageFetchers), speed)

  // Proactively keep the Claude session synced with Claude Code's own store
  // (Keychain/file) every 5 minutes, so a session left idle between requests
  // does not go stale from a token rotation that happened outside this
  // plugin (the `claude` CLI refreshing on its own, or another consumer).
  if (claudeTokens !== undefined) {
    const syncTimer = setInterval(() => {
      claudeTokens?.session().catch(() => {
        // Best-effort: TokenManager already surfaces failures via onRemoved.
      })
    }, 5 * 60_000)
    ctx.effect(() => () => { clearInterval(syncTimer) }, 'dsh-plugin-subscriptions: claude background sync timer')
  }

  // `tools` is optional (headless/minimal compositions may not mount it), so
  // registration waits for the service instead of injecting it at load.
  // x_search and video_generate follow the grok provider; image_generate
  // prefers the codex provider and falls back to grok.
  ctx.inject(['tools'], (toolsCtx) => {
    if (grokTokens !== undefined) {
      toolsCtx.tools.register(createXSearchTool({ tokens: grokTokens }))
      toolsCtx.tools.register(createVideoGenerateTool({ tokens: grokTokens }))
    }
    if (codexTokens !== undefined || grokTokens !== undefined) {
      toolsCtx.tools.register(createImageGenerateTool({
        ...codexTokens === undefined ? {} : { codexTokens },
        ...grokTokens === undefined ? {} : { grokTokens },
        resolveAttachments,
        resolveLlm: () => ctx.get('llm'),
      }))
    }
  })
}
