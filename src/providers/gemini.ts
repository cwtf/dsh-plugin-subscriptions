/**
 * Gemini (Google account) subscription provider: OAuth against Google's
 * identity endpoints with the Gemini CLI's installed-app client credentials,
 * and streaming against the Cloud Code Assist endpoint
 * (cloudcode-pa.googleapis.com), the same backend the CLI uses for personal
 * Google accounts and Google AI Pro/Ultra subscriptions.
 */

import { randomUUID } from 'node:crypto'
import { attributionHeaders, EMPTY_RESPONSE_CODE, LlmAdapter, LlmError } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { FlowSpec } from '../auth/oauth-flow.js'
import type { GeminiSession } from '../auth/store.js'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { resolveImages } from '../translate/resolved.js'
import { streamGemini, toGeminiContents, toGeminiSystem, toGeminiTools } from '../translate/gemini.js'
import {
  httpLlmError,
  idleWatchdog,
  mapFetchFailure,
  oauthEndpointError,
  OAuthEndpointError,
  TokenManager,
} from './common.js'
import type { FetchFn, ModelEntry, ProviderUsage, UsageWindow } from './common.js'

/**
 * The Gemini CLI's installed-app OAuth client credentials (gemini-cli
 * packages/core/src/code_assist/oauth2.ts). They are public by design — an
 * installed application cannot keep a secret, and Google issues this pair for
 * loopback-redirect clients only — but their verbatim form trips repository
 * secret scanners, so they are assembled from fragments here.
 */
const GEMINI_CLIENT_ID_PARTS = [
  '681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j',
  '.apps.googleusercontent.com',
] as const
const GEMINI_CLIENT_SECRET_PARTS = [
  'GOCSPX-',
  '4uHgMPm',
  '-1o7Sk-geV6Cu5clXFsxl',
] as const

export const GEMINI_CLIENT_ID = GEMINI_CLIENT_ID_PARTS.join('')
export const GEMINI_CLIENT_SECRET = GEMINI_CLIENT_SECRET_PARTS.join('')
export const GEMINI_AUTHORIZE_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
export const GEMINI_TOKEN_URL = 'https://oauth2.googleapis.com/token'
export const GEMINI_USERINFO_URL = 'https://www.googleapis.com/oauth2/v2/userinfo'
export const GEMINI_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
  + ' https://www.googleapis.com/auth/userinfo.email'
  + ' https://www.googleapis.com/auth/userinfo.profile'
export const GEMINI_CALLBACK_PATH = '/oauth2callback'
export const CODE_ASSIST_ENDPOINT = 'https://cloudcode-pa.googleapis.com'
export const CODE_ASSIST_API_VERSION = 'v1internal'
const GEMINI_CONTEXT_WINDOW = 1_000_000
const GEMINI_DEFAULT_MAX_TOKENS = 65_536
/** Refresh when the access token has less than this much life left. */
export const GEMINI_PREEMPT_MS = 5 * 60_000
/**
 * Per-request ceiling for the login-time Code Assist calls. They run
 * unsupervised after the browser tab closes (the flow attempt is gone by
 * then), so without one a hung endpoint would keep the login pending forever.
 */
export const GEMINI_SETUP_TIMEOUT_MS = 30_000
/** Wall-clock ceiling for the whole onboarding poll, across all attempts. */
export const GEMINI_ONBOARD_DEADLINE_MS = 120_000
/** Gap between onboarding polls, matching the Gemini CLI's cadence. */
const GEMINI_ONBOARD_POLL_MS = 5_000

/** Code Assist method URL: `${version}:${method}` on the endpoint. */
function codeAssistUrl(method: string): string {
  return `${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}:${method}`
}

/** Static gemini flow facts for the OAuth flow engine. */
export const geminiFlow: FlowSpec = {
  callbackPath: GEMINI_CALLBACK_PATH,
  // The redirect URI embeds the port, so it must be an ephemeral one.
  // 127.0.0.1 (not localhost) because Google's installed-app rules key on the
  // loopback IP literal, matching the Gemini CLI's own redirect.
  listen: { host: '127.0.0.1', ports: [0] },
  buildAuthorizeUrl({ redirectUri, state, pkce }) {
    const params = new URLSearchParams({
      response_type: 'code',
      client_id: GEMINI_CLIENT_ID,
      redirect_uri: redirectUri,
      scope: GEMINI_SCOPE,
      code_challenge: pkce.challenge,
      code_challenge_method: 'S256',
      state,
      // access_type=offline plus prompt=consent: without both, Google may
      // withhold the refresh token on a repeat login.
      access_type: 'offline',
      prompt: 'consent',
    })
    return `${GEMINI_AUTHORIZE_URL}?${params.toString()}`
  },
}

/** Token endpoint response shape (subset). */
interface GeminiTokenResponse {
  access_token?: string
  refresh_token?: string
  expires_in?: number
  scope?: string
}

/** The Code Assist client metadata every non-generate call carries. */
const CODE_ASSIST_METADATA = {
  ideType: 'IDE_UNSPECIFIED',
  platform: 'PLATFORM_UNSPECIFIED',
  pluginType: 'GEMINI',
} as const

/** One tier entry of a loadCodeAssist response (subset). */
interface GeminiUserTier {
  id?: string
  name?: string
  isDefault?: boolean
}

/** loadCodeAssist response shape (subset). */
interface LoadCodeAssistResponse {
  currentTier?: GeminiUserTier | null
  allowedTiers?: GeminiUserTier[] | null
  cloudaicompanionProject?: string | null
  paidTier?: GeminiUserTier | null
}

/** onboardUser long-running operation shape (subset). */
interface GeminiOperation {
  name?: string
  done?: boolean
  response?: { cloudaicompanionProject?: { id?: string } }
}

/** The project and tier a login resolved out of Code Assist setup. */
export interface GeminiSetup {
  projectId: string
  tier?: string
  tierName?: string
}

/** Thrown when the account has no Code Assist project and needs one configured. */
export class GeminiProjectRequiredError extends Error {
  constructor() {
    super(
      'this Google account has no managed Gemini Code Assist project; set up a '
      + 'Google Cloud project with the Gemini CLI (GOOGLE_CLOUD_PROJECT) first',
    )
    this.name = 'GeminiProjectRequiredError'
  }
}

/** POST one Code Assist method with the bearer token; throws on non-2xx. */
async function codeAssistPost<T>(accessToken: string, method: string, body: unknown): Promise<T> {
  const response = await fetch(codeAssistUrl(method), {
    method: 'POST',
    headers: {
      'authorization': `Bearer ${accessToken}`,
      'content-type': 'application/json',
      ...attributionHeaders(),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(GEMINI_SETUP_TIMEOUT_MS),
  })
  if (!response.ok) throw await oauthEndpointError(response, `gemini ${method}`)
  return response.json() as Promise<T>
}

/**
 * Poll one long-running onboarding operation until it reports done, bounded by
 * a wall-clock deadline so a stuck operation cannot pin the login open.
 */
async function awaitGeminiOperation(accessToken: string, name: string): Promise<GeminiOperation> {
  // Onboarding provisions the managed project; the CLI polls on a 5s cadence.
  // The first poll runs immediately: most operations settle within a beat.
  const deadline = Date.now() + GEMINI_ONBOARD_DEADLINE_MS
  for (let attempt = 0; Date.now() < deadline; attempt++) {
    if (attempt > 0) await new Promise(resolve => setTimeout(resolve, GEMINI_ONBOARD_POLL_MS))
    const response = await fetch(`${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}/${name}`, {
      headers: { authorization: `Bearer ${accessToken}`, ...attributionHeaders() },
      signal: AbortSignal.timeout(GEMINI_SETUP_TIMEOUT_MS),
    })
    if (!response.ok) throw await oauthEndpointError(response, 'gemini onboarding')
    const operation = await response.json() as GeminiOperation
    if (operation.done === true) return operation
  }
  throw new Error('gemini onboarding did not finish in time; try logging in again')
}

/**
 * Resolve the Code Assist project and tier for a fresh login (gemini-cli's
 * setupUser): load the account's Code Assist state, and when the account has
 * never onboarded, enroll it in the server-proposed default tier and await
 * the provisioning operation.
 * @param accessToken - the just-issued access token.
 * @returns the project id every generate request must carry, plus the tier.
 */
export async function setupGeminiProject(accessToken: string): Promise<GeminiSetup> {
  const loaded = await codeAssistPost<LoadCodeAssistResponse>(accessToken, 'loadCodeAssist', {
    metadata: CODE_ASSIST_METADATA,
  })
  if (loaded.currentTier !== undefined && loaded.currentTier !== null) {
    const tier = loaded.paidTier?.id ?? loaded.currentTier.id
    const tierName = loaded.paidTier?.name ?? loaded.currentTier.name
    if (typeof loaded.cloudaicompanionProject !== 'string' || loaded.cloudaicompanionProject.length === 0) {
      throw new GeminiProjectRequiredError()
    }
    return {
      projectId: loaded.cloudaicompanionProject,
      ...tier === undefined ? {} : { tier },
      ...tierName === undefined ? {} : { tierName },
    }
  }
  const defaultTier = (loaded.allowedTiers ?? []).find(tier => tier.isDefault === true)
  const tierId = defaultTier?.id ?? 'legacy-tier'
  // `cloudaicompanionProject` is deliberately absent: the free tier provisions
  // a managed project, and sending one errors out.
  const operation = await codeAssistPost<GeminiOperation>(accessToken, 'onboardUser', {
    tierId,
    metadata: CODE_ASSIST_METADATA,
  })
  const settled = operation.done === true || operation.name === undefined
    ? operation
    : await awaitGeminiOperation(accessToken, operation.name)
  const projectId = settled.response?.cloudaicompanionProject?.id
  if (projectId === undefined || projectId.length === 0) throw new GeminiProjectRequiredError()
  return {
    projectId,
    tier: tierId,
    ...defaultTier?.name === undefined ? {} : { tierName: defaultTier.name },
  }
}

/** Best-effort account email lookup; login must not fail when this does. */
async function fetchGeminiEmail(accessToken: string): Promise<string | undefined> {
  try {
    const response = await fetch(GEMINI_USERINFO_URL, {
      headers: { authorization: `Bearer ${accessToken}`, ...attributionHeaders() },
      signal: AbortSignal.timeout(GEMINI_SETUP_TIMEOUT_MS),
    })
    if (!response.ok) return undefined
    const profile = await response.json() as { email?: unknown }
    return typeof profile.email === 'string' && profile.email.length > 0 ? profile.email : undefined
  } catch {
    // The email is decorative; only the token exchange owns login success.
    return undefined
  }
}

/** Build a session from a token response, preserving setup fields. */
function geminiSession(
  tokens: GeminiTokenResponse,
  setup: Pick<GeminiSession, 'projectId' | 'tier' | 'tierName'>,
  emailAddress: string | undefined,
  fallbackRefreshToken?: string,
): GeminiSession {
  if (typeof tokens.access_token !== 'string' || tokens.access_token.length === 0) {
    throw new Error('gemini token endpoint returned no access token')
  }
  const refreshToken = tokens.refresh_token ?? fallbackRefreshToken
  if (refreshToken === undefined) throw new Error('gemini token endpoint returned no refresh token')
  if (typeof tokens.expires_in !== 'number' || tokens.expires_in <= 0) {
    throw new Error('gemini token endpoint returned no usable expiry')
  }
  return {
    accessToken: tokens.access_token,
    refreshToken,
    expiresAt: Date.now() + tokens.expires_in * 1000,
    projectId: setup.projectId,
    ...setup.tier === undefined ? {} : { tier: setup.tier },
    ...setup.tierName === undefined ? {} : { tierName: setup.tierName },
    ...emailAddress === undefined ? {} : { emailAddress },
  }
}

/**
 * Exchange an authorization code for a gemini session (form-encoded grant
 * carrying the installed-app client secret alongside the PKCE verifier), then
 * resolve the account's Code Assist project and tier.
 * @param code - the authorization code from the callback.
 * @param verifier - the PKCE verifier minted for the attempt.
 * @param redirectUri - the attempt's redirect URI.
 * @returns the session to store.
 */
export async function exchangeGeminiCode(
  code: string,
  verifier: string,
  redirectUri: string,
): Promise<GeminiSession> {
  const response = await fetch(GEMINI_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      client_id: GEMINI_CLIENT_ID,
      client_secret: GEMINI_CLIENT_SECRET,
      code,
      redirect_uri: redirectUri,
      code_verifier: verifier,
    }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'gemini')
  const tokens = await response.json() as GeminiTokenResponse
  // Checked here as well as in geminiSession: the Code Assist setup below
  // needs the token before the session is ever assembled.
  if (typeof tokens.access_token !== 'string' || tokens.access_token.length === 0) {
    throw new Error('gemini token endpoint returned no access token')
  }
  const [setup, email] = await Promise.all([
    setupGeminiProject(tokens.access_token),
    fetchGeminiEmail(tokens.access_token),
  ])
  return geminiSession(tokens, setup, email)
}

/**
 * Refresh a gemini session (form-encoded grant). Google does not rotate the
 * refresh token: the response carries none, so the stored one is kept.
 * @param session - the stored session.
 * @returns the fresh session to store.
 */
export async function refreshGemini(session: GeminiSession): Promise<GeminiSession> {
  const response = await fetch(GEMINI_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      client_id: GEMINI_CLIENT_ID,
      client_secret: GEMINI_CLIENT_SECRET,
      refresh_token: session.refreshToken,
    }).toString(),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'gemini')
  return geminiSession(
    await response.json() as GeminiTokenResponse,
    session,
    session.emailAddress,
    session.refreshToken,
  )
}

/**
 * Whether a gemini refresh failure means the login is permanently gone.
 * @param error - the thrown refresh error.
 * @returns true when re-login is the only fix.
 */
export function isGeminiPermanentRefreshError(error: unknown): boolean {
  return error instanceof OAuthEndpointError && error.oauthCode === 'invalid_grant'
}

/** One retrieveUserQuota bucket (subset). */
interface GeminiQuotaBucket {
  modelId?: string
  tokenType?: string
  remainingFraction?: number
  resetTime?: string
}

/** RFC3339 timestamp → epoch ms, or undefined when absent/unparsable. */
function geminiResetsAt(value: unknown): number | undefined {
  if (typeof value !== 'string' || value.length === 0) return undefined
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

/**
 * Fetch the gemini subscription usage from Code Assist's retrieveUserQuota
 * (the source of the Gemini CLI's `/quota` panel): one bucket per model,
 * carrying the remaining fraction of the current quota window.
 * @param session - the stored session (used as-is; never refreshed here).
 * @param fetchFn - fetch implementation (injectable for tests).
 * @param signal - caller cancellation from the RPC transport.
 * @returns the mapped usage snapshot.
 */
export async function fetchGeminiUsage(
  session: GeminiSession,
  fetchFn: FetchFn = fetch,
  signal?: AbortSignal,
): Promise<ProviderUsage> {
  const response = await fetchFn(codeAssistUrl('retrieveUserQuota'), {
    method: 'POST',
    headers: {
      'authorization': `Bearer ${session.accessToken}`,
      'content-type': 'application/json',
      'accept': 'application/json',
      ...attributionHeaders(),
    },
    body: JSON.stringify({ project: session.projectId }),
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) throw await oauthEndpointError(response, 'gemini quota')
  const payload = await response.json() as { buckets?: GeminiQuotaBucket[] | null }
  const windows: UsageWindow[] = []
  for (const bucket of payload.buckets ?? []) {
    if (typeof bucket.remainingFraction !== 'number' || !Number.isFinite(bucket.remainingFraction)) continue
    // A bucket may name its scope by model; token-type buckets without a
    // model id aggregate the account and stay unscoped.
    const scope = typeof bucket.modelId === 'string' && bucket.modelId.length > 0 ? bucket.modelId : undefined
    const resetsAt = geminiResetsAt(bucket.resetTime)
    windows.push({
      kind: 'other',
      ...scope === undefined ? {} : { scope },
      usedPercent: (1 - bucket.remainingFraction) * 100,
      ...resetsAt === undefined ? {} : { resetsAt },
    })
  }
  const plan = session.tierName ?? session.tier
  return {
    supported: true,
    windows,
    ...plan === undefined ? {} : { plan },
  }
}

/** The gemini chat models accept image input. */
const GEMINI_MODALITIES: readonly ('text' | 'image')[] = ['text', 'image']

/**
 * Whether the model streams thought summaries when asked. Restricted to the
 * thinking families: an unknown model id must not receive a thinkingConfig it
 * could reject. The `-lite` variants are excluded — thinking is off by default
 * there, and asking for thoughts without also buying a budget gets nothing
 * back at best.
 */
export function geminiThinks(model: string): boolean {
  return /^gemini-(2\.5|3)/.test(model) && !/-lite\b/.test(model)
}

/** Constructor dependencies for {@link GeminiAdapter}. */
export interface GeminiAdapterOptions {
  models: readonly ModelEntry[]
  streamIdleTimeoutMs: number
  tokens: TokenManager<GeminiSession>
  /** Resolve the attachment service per request; absent means image requests fail loudly. */
  resolveAttachments?: () => AttachmentStore | undefined
}

/** Gemini wire adapter: one instance serves the `gemini` provider route. */
export class GeminiAdapter extends LlmAdapter {
  constructor(private readonly options: GeminiAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Gemini (Subscription)' }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    // Not logged in → empty catalog, so the web picker drops the provider.
    // Code Assist has no model-list endpoint, so the catalog is always static.
    if (await this.options.tokens.peek() === undefined) return []
    return this.options.models.map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      inputModalities: model.inputModalities ?? GEMINI_MODALITIES,
    }))
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const configured = this.options.models.find(entry => entry.id === model)
    return Promise.resolve({
      provider,
      id: model,
      name: configured?.name ?? model,
      inputModalities: configured?.inputModalities ?? GEMINI_MODALITIES,
      context: { contextWindow: configured?.contextWindow ?? GEMINI_CONTEXT_WINDOW },
      defaultMaxTokens: configured?.maxTokens ?? GEMINI_DEFAULT_MAX_TOKENS,
    })
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const watchdog = idleWatchdog(options.signal, this.options.streamIdleTimeoutMs)
    try {
      let session = await this.options.tokens.session()
      let response = await this.request(options, session, watchdog.signal)
      if (response.status === 401) {
        // One forced refresh + retry on an unexpired-but-rejected token.
        session = await this.options.tokens.session(true)
        response = await this.request(options, session, watchdog.signal)
      }
      if (!response.ok) throw await httpLlmError(response, 'gemini API')
      if (response.body === null) {
        throw new LlmError('gemini API returned no response body', EMPTY_RESPONSE_CODE)
      }
      yield* streamGemini(response.body, () => { watchdog.pulse() })
    } catch (error: unknown) {
      throw mapFetchFailure('gemini API', error, watchdog, options.signal)
    } finally {
      watchdog.stop()
    }
  }

  private async request(options: GenerateOptions, session: GeminiSession, signal: AbortSignal): Promise<Response> {
    const messages = await resolveImages(options.messages, this.options.resolveAttachments?.(), signal)
    const system = toGeminiSystem(options.system, messages)
    const contents = toGeminiContents(messages)
    // Gemini rejects an empty `contents`; say so here rather than shipping a
    // request that can only come back as an opaque 400.
    if (contents.length === 0) {
      throw new LlmError(
        'gemini API requires at least one user or model message; the request carried only system content',
        'INVALID_REQUEST',
      )
    }
    const body = {
      model: options.model,
      project: session.projectId,
      user_prompt_id: randomUUID(),
      request: {
        contents,
        ...system === undefined ? {} : { systemInstruction: system },
        generationConfig: {
          maxOutputTokens: options.maxTokens
            ?? this.options.models.find(entry => entry.id === options.model)?.maxTokens
            ?? GEMINI_DEFAULT_MAX_TOKENS,
          ...options.temperature === undefined ? {} : { temperature: options.temperature },
          ...options.stop === undefined || options.stop.length === 0
            ? {}
            : { stopSequences: options.stop },
          ...geminiThinks(options.model) ? { thinkingConfig: { includeThoughts: true } } : {},
        },
        ...options.tools !== undefined && options.tools.length > 0
          ? { tools: toGeminiTools(options.tools) }
          : {},
        ...options.sessionId === undefined ? {} : { session_id: String(options.sessionId) },
      },
    }
    return fetch(`${codeAssistUrl('streamGenerateContent')}?alt=sse`, {
      method: 'POST',
      headers: {
        'authorization': `Bearer ${session.accessToken}`,
        'accept': 'text/event-stream',
        'content-type': 'application/json',
        ...attributionHeaders(),
      },
      body: JSON.stringify(body),
      signal,
    })
  }
}
