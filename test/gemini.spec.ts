/**
 * Gemini provider: the OAuth flow facts and token exchange (via a routed
 * fake `fetch`, no network), Code Assist project setup and quota mapping, and
 * the Gemini wire translator's request assembly and SSE state machine.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { CallId, LlmError, MessageId, QUOTA_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, StreamChunk, ToolCallBlock } from '@deepseek-ai/dsh-llm'

import { OAuthFlowManager } from '../src/auth/oauth-flow.js'
import {
  CODE_ASSIST_ENDPOINT,
  exchangeGeminiCode,
  fetchGeminiUsage,
  GEMINI_AUTHORIZE_URL,
  GEMINI_CALLBACK_PATH,
  GEMINI_CLIENT_ID,
  GEMINI_CLIENT_SECRET,
  GEMINI_SCOPE,
  GEMINI_TOKEN_URL,
  GEMINI_USERINFO_URL,
  GeminiAdapter,
  GeminiProjectRequiredError,
  geminiFlow,
  geminiThinks,
  isGeminiPermanentRefreshError,
  refreshGemini,
} from '../src/providers/gemini.js'
import type { GeminiSession } from '../src/auth/store.js'
import { OAuthEndpointError, TokenManager } from '../src/providers/common.js'
import type { FetchFn } from '../src/providers/common.js'
import {
  GeminiStreamTranslator,
  mapGeminiUsage,
  sanitizeGeminiSchema,
  streamGemini,
  toGeminiContents,
  toGeminiSystem,
  toGeminiTools,
} from '../src/translate/gemini.js'
import type { GeminiStreamEvent } from '../src/translate/gemini.js'

// ---------------------------------------------------------------------------
// The constants the authorize request is built from
// ---------------------------------------------------------------------------

test('Gemini OAuth parameters match what the Gemini CLI sends', () => {
  // Literals on purpose for the endpoints, mirroring the Claude parameter
  // test in login.spec.ts. The client credentials are asserted structurally
  // instead: their verbatim values trip repository secret scanners (they are
  // the Gemini CLI's public installed-app pair, not leaked secrets).
  assert.equal(GEMINI_AUTHORIZE_URL, 'https://accounts.google.com/o/oauth2/v2/auth')
  assert.equal(GEMINI_TOKEN_URL, 'https://oauth2.googleapis.com/token')
  assert.match(GEMINI_CLIENT_ID, /^\d{12}-[\da-z]{32}\.apps\.googleusercontent\.com$/)
  assert.match(GEMINI_CLIENT_SECRET, /^GOCSPX-[\w-]{28}$/)
  assert.equal(GEMINI_CALLBACK_PATH, '/oauth2callback')
  assert.equal(
    GEMINI_SCOPE,
    'https://www.googleapis.com/auth/cloud-platform'
    + ' https://www.googleapis.com/auth/userinfo.email'
    + ' https://www.googleapis.com/auth/userinfo.profile',
  )
})

test('geminiFlow builds the CLI\'s authorize URL on a loopback IP', async () => {
  const flows = new OAuthFlowManager()
  const attempt = await flows.start('gemini', geminiFlow)
  // Consume the code promise: the test cancels the attempt, and an unowned
  // rejection would surface as an unhandledRejection after the test ends.
  void attempt.waitCode().catch(() => undefined)
  try {
    const url = new URL(attempt.authorizeUrl)
    assert.equal(`${url.origin}${url.pathname}`, GEMINI_AUTHORIZE_URL)
    const params = url.searchParams
    assert.equal(params.get('response_type'), 'code')
    assert.equal(params.get('client_id'), GEMINI_CLIENT_ID)
    assert.equal(params.get('scope'), GEMINI_SCOPE)
    assert.equal(params.get('code_challenge_method'), 'S256')
    assert.ok(params.get('code_challenge'), 'PKCE challenge present')
    // Without both, Google withholds the refresh token on a repeat login.
    assert.equal(params.get('access_type'), 'offline')
    assert.equal(params.get('prompt'), 'consent')

    const redirectUri = new URL(params.get('redirect_uri') ?? '')
    assert.equal(redirectUri.protocol, 'http:')
    assert.equal(redirectUri.hostname, '127.0.0.1', 'the redirect keys on the loopback IP literal')
    assert.ok(Number(redirectUri.port) > 0, 'the redirect URI carries the listening port')
    assert.equal(redirectUri.pathname, GEMINI_CALLBACK_PATH)
  } finally {
    attempt.cancel()
  }
})

// ---------------------------------------------------------------------------
// Token exchange + Code Assist setup (fake fetch routed by URL)
// ---------------------------------------------------------------------------

/** Recorded request of the fake fetch below. */
interface SeenRequest {
  url: string
  method: string
  body: string | undefined
}

/**
 * Replace global fetch with a URL-routed fake. Routes map a URL to the JSON
 * payload it answers; unlisted URLs 404. Returns the request log and the
 * restore function.
 */
function fakeRoutes(routes: Record<string, unknown>): { seen: SeenRequest[]; restore: () => void } {
  const real = globalThis.fetch
  const seen: SeenRequest[] = []
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)
    seen.push({
      url,
      method: init?.method ?? 'GET',
      body: typeof init?.body === 'string' ? init.body : undefined,
    })
    if (!(url in routes)) return new Response('not found', { status: 404 })
    return Response.json(routes[url])
  }) as typeof fetch
  return { seen, restore: () => { globalThis.fetch = real } }
}

const LOAD_CODE_ASSIST_URL = `${CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist`
const ONBOARD_USER_URL = `${CODE_ASSIST_ENDPOINT}/v1internal:onboardUser`

test('exchangeGeminiCode: token grant, currentTier setup, and email lookup', async () => {
  const fake = fakeRoutes({
    [GEMINI_TOKEN_URL]: { access_token: 'gem-at', refresh_token: 'gem-rt', expires_in: 3600 },
    [LOAD_CODE_ASSIST_URL]: {
      currentTier: { id: 'free-tier', name: 'Gemini Code Assist for individuals' },
      cloudaicompanionProject: 'managed-project-1',
    },
    [GEMINI_USERINFO_URL]: { email: 'user@example.com' },
  })
  try {
    const session = await exchangeGeminiCode('the-code', 'the-verifier', 'http://127.0.0.1:1/oauth2callback')
    assert.equal(session.accessToken, 'gem-at')
    assert.equal(session.refreshToken, 'gem-rt')
    assert.equal(session.projectId, 'managed-project-1')
    assert.equal(session.tier, 'free-tier')
    assert.equal(session.tierName, 'Gemini Code Assist for individuals')
    assert.equal(session.emailAddress, 'user@example.com')
    assert.ok(session.expiresAt > Date.now())

    const tokenRequest = fake.seen.find(request => request.url === GEMINI_TOKEN_URL)
    assert.ok(tokenRequest !== undefined)
    const params = new URLSearchParams(tokenRequest.body)
    assert.equal(params.get('grant_type'), 'authorization_code')
    assert.equal(params.get('client_id'), GEMINI_CLIENT_ID)
    assert.equal(params.get('client_secret'), GEMINI_CLIENT_SECRET)
    assert.equal(params.get('code'), 'the-code')
    assert.equal(params.get('code_verifier'), 'the-verifier')
    assert.equal(params.get('redirect_uri'), 'http://127.0.0.1:1/oauth2callback')

    const loadRequest = fake.seen.find(request => request.url === LOAD_CODE_ASSIST_URL)
    assert.ok(loadRequest !== undefined, 'the login resolved the Code Assist project')
  } finally {
    fake.restore()
  }
})

test('exchangeGeminiCode: a new account onboards onto the default tier', async () => {
  const fake = fakeRoutes({
    [GEMINI_TOKEN_URL]: { access_token: 'gem-at', refresh_token: 'gem-rt', expires_in: 3600 },
    [LOAD_CODE_ASSIST_URL]: {
      currentTier: null,
      allowedTiers: [{ id: 'free-tier', name: 'Free', isDefault: true }],
    },
    [ONBOARD_USER_URL]: {
      done: true,
      response: { cloudaicompanionProject: { id: 'provisioned-1' } },
    },
    [GEMINI_USERINFO_URL]: { email: 'user@example.com' },
  })
  try {
    const session = await exchangeGeminiCode('the-code', 'the-verifier', 'http://127.0.0.1:1/oauth2callback')
    assert.equal(session.projectId, 'provisioned-1')
    assert.equal(session.tier, 'free-tier')
    const onboardRequest = fake.seen.find(request => request.url === ONBOARD_USER_URL)
    assert.ok(onboardRequest !== undefined)
    const body = JSON.parse(onboardRequest.body ?? '{}') as { tierId?: string }
    assert.equal(body.tierId, 'free-tier')
  } finally {
    fake.restore()
  }
})

test('exchangeGeminiCode: onboarding polls the long-running operation', async () => {
  const operationUrl = `${CODE_ASSIST_ENDPOINT}/v1internal/operations/op-1`
  const fake = fakeRoutes({
    [GEMINI_TOKEN_URL]: { access_token: 'gem-at', refresh_token: 'gem-rt', expires_in: 3600 },
    [LOAD_CODE_ASSIST_URL]: { currentTier: null, allowedTiers: [] },
    [ONBOARD_USER_URL]: { done: false, name: 'operations/op-1' },
    [operationUrl]: { done: true, response: { cloudaicompanionProject: { id: 'provisioned-2' } } },
    [GEMINI_USERINFO_URL]: {},
  })
  try {
    const session = await exchangeGeminiCode('the-code', 'the-verifier', 'http://127.0.0.1:1/oauth2callback')
    assert.equal(session.projectId, 'provisioned-2')
    assert.equal(session.tier, 'legacy-tier', 'no default tier falls back to legacy-tier')
    assert.ok(fake.seen.some(request => request.url === operationUrl), 'the operation was polled')
    assert.equal(session.emailAddress, undefined, 'a userinfo miss is decorative, not fatal')
  } finally {
    fake.restore()
  }
})

test('exchangeGeminiCode: an account without a project fails loud', async () => {
  const fake = fakeRoutes({
    [GEMINI_TOKEN_URL]: { access_token: 'gem-at', refresh_token: 'gem-rt', expires_in: 3600 },
    [LOAD_CODE_ASSIST_URL]: { currentTier: { id: 'standard-tier' }, cloudaicompanionProject: null },
    [GEMINI_USERINFO_URL]: {},
  })
  try {
    await assert.rejects(
      exchangeGeminiCode('the-code', 'the-verifier', 'http://127.0.0.1:1/oauth2callback'),
      (error: unknown) => error instanceof GeminiProjectRequiredError,
    )
  } finally {
    fake.restore()
  }
})

test('exchangeGeminiCode: a refused token exchange surfaces the OAuth error', async () => {
  const real = globalThis.fetch
  globalThis.fetch = (async () => Response.json(
    { error: 'invalid_grant', error_description: 'code expired' },
    { status: 400 },
  )) as typeof fetch
  try {
    await assert.rejects(
      exchangeGeminiCode('stale', 'the-verifier', 'http://127.0.0.1:1/oauth2callback'),
      /gemini token endpoint error \(HTTP 400\): code expired/,
    )
  } finally {
    globalThis.fetch = real
  }
})

test('refreshGemini keeps the refresh token and setup fields Google omits', async () => {
  const stored: GeminiSession = {
    accessToken: 'old-at',
    refreshToken: 'keep-rt',
    expiresAt: Date.now() - 1_000,
    projectId: 'managed-project-1',
    tier: 'free-tier',
    tierName: 'Free',
    emailAddress: 'user@example.com',
  }
  const fake = fakeRoutes({
    // Google answers a refresh with a bare access token: no refresh_token.
    [GEMINI_TOKEN_URL]: { access_token: 'new-at', expires_in: 3600 },
  })
  try {
    const next = await refreshGemini(stored)
    assert.equal(next.accessToken, 'new-at')
    assert.equal(next.refreshToken, 'keep-rt', 'the unrotated refresh token survives')
    assert.equal(next.projectId, 'managed-project-1')
    assert.equal(next.tier, 'free-tier')
    assert.equal(next.tierName, 'Free')
    assert.equal(next.emailAddress, 'user@example.com')
    const params = new URLSearchParams(fake.seen[0].body)
    assert.equal(params.get('grant_type'), 'refresh_token')
    assert.equal(params.get('refresh_token'), 'keep-rt')
    assert.equal(params.get('client_secret'), GEMINI_CLIENT_SECRET)
  } finally {
    fake.restore()
  }
})

test('isGeminiPermanentRefreshError: invalid_grant is permanent, the rest transient', () => {
  assert.equal(isGeminiPermanentRefreshError(new OAuthEndpointError('x', 400, 'invalid_grant')), true)
  assert.equal(isGeminiPermanentRefreshError(new OAuthEndpointError('x', 500)), false)
  assert.equal(isGeminiPermanentRefreshError(new Error('network')), false)
})

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

const geminiSession: GeminiSession = {
  accessToken: 'at',
  refreshToken: 'rt',
  expiresAt: Date.now() + 3_600_000,
  projectId: 'managed-project-1',
  tier: 'free-tier',
  tierName: 'Gemini Code Assist for individuals',
}

test('fetchGeminiUsage maps quota buckets to windows with model scopes', async () => {
  const requests: { url: string; body: string | undefined; headers: Record<string, string> }[] = []
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    const headers: Record<string, string> = {}
    new Headers(init?.headers).forEach((value, key) => { headers[key] = value })
    requests.push({ url: String(url), body: typeof init?.body === 'string' ? init.body : undefined, headers })
    return Response.json({
      buckets: [
        { modelId: 'gemini-2.5-pro', tokenType: 'REQUESTS', remainingFraction: 0.75, resetTime: '2026-08-24T00:00:00Z' },
        { modelId: 'gemini-2.5-flash', tokenType: 'REQUESTS', remainingFraction: 1 },
        { tokenType: 'REQUESTS' }, // no fraction → skipped
      ],
    })
  }) as FetchFn
  const usage = await fetchGeminiUsage(geminiSession, fetchFn)
  assert.deepEqual(usage, {
    supported: true,
    plan: 'Gemini Code Assist for individuals',
    windows: [
      { kind: 'other', scope: 'gemini-2.5-pro', usedPercent: 25, resetsAt: Date.parse('2026-08-24T00:00:00Z') },
      { kind: 'other', scope: 'gemini-2.5-flash', usedPercent: 0 },
    ],
  })
  assert.equal(requests.length, 1)
  assert.match(requests[0].url, /v1internal:retrieveUserQuota/)
  assert.equal(requests[0].headers.authorization, 'Bearer at')
  assert.deepEqual(JSON.parse(requests[0].body ?? '{}'), { project: 'managed-project-1' })
})

test('fetchGeminiUsage: non-2xx response throws', async () => {
  const fetchFn = (async () => new Response('nope', { status: 403 })) as FetchFn
  await assert.rejects(fetchGeminiUsage(geminiSession, fetchFn), /gemini quota/)
})

// ---------------------------------------------------------------------------
// Request translation
// ---------------------------------------------------------------------------

let callCounter = 0

function toolCall(name: string, args: string): ToolCallBlock {
  return { type: 'tool-call', id: CallId(`call-${++callCounter}`), name, arguments: args }
}

function toolResult(callId: string, text: string): ContentBlock {
  return { type: 'tool-result', toolCallId: CallId(callId), content: [{ type: 'text', text }] }
}

test('toGeminiContents: roles, merge, tool calls, and results', () => {
  const call = toolCall('bash', '{"cmd":"ls"}')
  const contents = toGeminiContents([
    { role: 'system', content: [{ type: 'text', text: 'system text' }] },
    { role: 'user', content: [{ type: 'text', text: 'list files' }] },
    { role: 'user', content: [{ type: 'text', text: 'now please' }] },
    {
      role: 'assistant',
      content: [{ type: 'text', text: 'running ls' }, call, { type: 'reasoning', text: 'not replayed' }],
    },
    { role: 'user', content: [toolResult(String(call.id), 'file-a\nfile-b')] },
  ])
  assert.deepEqual(contents, [
    { role: 'user', parts: [{ text: 'list files' }, { text: 'now please' }] },
    {
      role: 'model',
      parts: [
        { text: 'running ls' },
        { functionCall: { id: String(call.id), name: 'bash', args: { cmd: 'ls' } } },
      ],
    },
    {
      role: 'user',
      parts: [{
        // The result names its function through the matching call: the wire
        // correlates by name when the model issued no id.
        functionResponse: { id: String(call.id), name: 'bash', response: { output: 'file-a\nfile-b' } },
      }],
    },
  ])
})

test('toGeminiContents: images inline and unresolved ones skip', () => {
  const contents = toGeminiContents([{
    role: 'user',
    content: [
      { type: 'text', text: 'what is this?' },
      { type: 'image', mediaType: 'image/png', dataBase64: 'aGk=' },
    ],
  }])
  assert.deepEqual(contents, [{
    role: 'user',
    parts: [{ text: 'what is this?' }, { inlineData: { mimeType: 'image/png', data: 'aGk=' } }],
  }])
  const unresolved = toGeminiContents([{
    role: 'user',
    content: [{ type: 'image', attachment: { attachmentId: 'x' } } as never],
  }])
  assert.deepEqual(unresolved, [])
})

test('toGeminiContents: malformed tool-call arguments become an empty object', () => {
  const call = toolCall('bash', '{nope')
  const contents = toGeminiContents([{ role: 'assistant', content: [call] }])
  assert.deepEqual(contents[0].parts, [{
    functionCall: { id: String(call.id), name: 'bash', args: {} },
  }])
})

test('toGeminiSystem: explicit prompt, then system-role messages', () => {
  const both = toGeminiSystem('explicit', [
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    { role: 'system', content: [{ type: 'text', text: 'from history' }] },
  ])
  assert.deepEqual(both, { parts: [{ text: 'explicit' }, { text: 'from history' }] })
  assert.equal(toGeminiSystem(undefined, [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]), undefined)
})

test('toGeminiTools maps to one functionDeclarations entry', () => {
  assert.deepEqual(toGeminiTools([{ name: 'bash', description: 'run', parameters: { type: 'object' } }]), [
    { functionDeclarations: [{ name: 'bash', description: 'run', parameters: { type: 'object' } }] },
  ])
})

test('toGeminiTools strips the JSON Schema keywords Gemini rejects', () => {
  // This plugin's own tools declare `additionalProperties: false`, which
  // Gemini answers with 400 INVALID_ARGUMENT ("Unknown name ...").
  const [entry] = toGeminiTools([{
    name: 'x_search',
    description: 'search',
    parameters: {
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'object',
      additionalProperties: false,
      required: ['query'],
      properties: {
        query: { type: 'string', description: 'the query', minLength: 1 },
        limit: { type: 'integer', maximum: 50, multipleOf: 5 },
        nested: {
          type: 'object',
          additionalProperties: false,
          properties: { deep: { type: 'string', format: 'uri' } },
        },
        items: { type: 'array', items: { type: 'string', additionalProperties: false } },
      },
    },
  }])
  const declarations = (entry as { functionDeclarations: { parameters: unknown }[] }).functionDeclarations
  assert.deepEqual(declarations[0].parameters, {
    type: 'object',
    required: ['query'],
    properties: {
      query: { type: 'string', description: 'the query', minLength: 1 },
      // `multipleOf` has no Gemini equivalent and is dropped with the rest.
      limit: { type: 'integer', maximum: 50 },
      // Sanitizing recurses: nested objects and array items are cleaned too.
      nested: { type: 'object', properties: { deep: { type: 'string' } } },
      items: { type: 'array', items: { type: 'string' } },
    },
  })
})

test('sanitizeGeminiSchema: const, union types, and oneOf are remapped not dropped', () => {
  assert.deepEqual(sanitizeGeminiSchema({ const: 'fixed' }), { enum: ['fixed'] })
  assert.deepEqual(
    sanitizeGeminiSchema({ type: ['string', 'null'], description: 'maybe' }),
    { type: 'string', description: 'maybe', nullable: true },
  )
  assert.deepEqual(
    sanitizeGeminiSchema({ oneOf: [{ type: 'string' }, { type: 'number' }] }),
    { anyOf: [{ type: 'string' }, { type: 'number' }] },
  )
  // A recognized format survives; an unrecognized one does not.
  assert.deepEqual(sanitizeGeminiSchema({ type: 'string', format: 'date-time' }), {
    type: 'string', format: 'date-time',
  })
  assert.deepEqual(sanitizeGeminiSchema({ type: 'string', format: 'email' }), { type: 'string' })
  assert.deepEqual(sanitizeGeminiSchema('not an object'), {})
})

test('toGeminiContents: a failed tool result uses the error key', () => {
  const call = toolCall('bash', '{}')
  const contents = toGeminiContents([
    { role: 'assistant', content: [call] },
    {
      role: 'user',
      content: [{
        type: 'tool-result',
        toolCallId: call.id,
        content: [{ type: 'text', text: 'command not found' }],
        isError: true,
      }],
    },
  ])
  assert.deepEqual(contents[1].parts, [{
    functionResponse: { id: String(call.id), name: 'bash', response: { error: 'command not found' } },
  }])
})

test('toGeminiContents: thought signatures are replayed onto their parts', () => {
  const call = toolCall('bash', '{}')
  const contents = toGeminiContents([{
    role: 'assistant',
    content: [{ type: 'text', text: 'calling' }, call],
    source: {
      kind: 'model',
      provider: 'gemini',
      model: 'gemini-3-pro-preview',
      replayState: { response: {}, blocks: [{}, { thoughtSignature: 'sig-abc' }] },
    },
  }])
  assert.deepEqual(contents[0].parts, [
    { text: 'calling' },
    {
      functionCall: { id: String(call.id), name: 'bash', args: {} },
      // Gemini 3 rejects a replayed function call whose signature is missing.
      thoughtSignature: 'sig-abc',
    },
  ])
})

test('toGeminiContents: a replay envelope that no longer lines up is ignored', () => {
  const call = toolCall('bash', '{}')
  const contents = toGeminiContents([{
    role: 'assistant',
    content: [{ type: 'text', text: 'calling' }, call],
    source: {
      kind: 'model',
      provider: 'gemini',
      model: 'gemini-3-pro-preview',
      // One entry for two blocks: assembly dropped a block, so the envelope no
      // longer describes this content and must not be applied by position.
      replayState: { response: {}, blocks: [{ thoughtSignature: 'sig-abc' }] },
    },
  }])
  assert.deepEqual(contents[0].parts, [
    { text: 'calling' },
    { functionCall: { id: String(call.id), name: 'bash', args: {} } },
  ])
})

// ---------------------------------------------------------------------------
// Stream translation
// ---------------------------------------------------------------------------

/** Feed every event through a translator, then finalize, and flatten. */
function drain(translator: GeminiStreamTranslator, events: GeminiStreamEvent[]): StreamChunk[] {
  return [...events.flatMap(event => translator.push(event)), ...translator.finish()]
}

test('gemini stream: text deltas, usage, and a stop finish', () => {
  const chunks = drain(new GeminiStreamTranslator(), [
    { response: { candidates: [{ content: { parts: [{ text: 'Hel' }] } }] } },
    { response: { candidates: [{ content: { parts: [{ text: 'lo' }] } }] } },
    {
      response: {
        candidates: [{ content: { parts: [] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 2, cachedContentTokenCount: 5, thoughtsTokenCount: 3 },
      },
    },
  ])
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hel' },
    { type: 'text-delta', index: 0, text: 'lo' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
    {
      type: 'usage',
      usage: { inputTokens: 7, outputTokens: 5, cacheReadTokens: 5, reasoningTokens: 3 },
    },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('gemini stream: thought parts open a reasoning block, then text follows', () => {
  const chunks = drain(new GeminiStreamTranslator(), [
    { response: { candidates: [{ content: { parts: [{ text: 'thinking ', thought: true }] } }] } },
    { response: { candidates: [{ content: { parts: [{ text: 'hard', thought: true }] } }] } },
    { response: { candidates: [{ content: { parts: [{ text: 'answer' }] }, finishReason: 'STOP' }] } },
  ])
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'thinking ' },
    { type: 'reasoning-delta', index: 0, text: 'hard' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'thinking hard' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: 'answer' },
    { type: 'block-end', index: 1, block: { type: 'text', text: 'answer' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('gemini stream: a functionCall part is one atomic tool-call block', () => {
  const chunks = drain(new GeminiStreamTranslator(() => 'call-0'), [
    {
      response: {
        candidates: [{
          content: { parts: [{ functionCall: { name: 'bash', args: { cmd: 'ls' } } }] },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 4 },
      },
    },
  ])
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 0, id: CallId('call-0'), name: 'bash', argumentsDelta: '{"cmd":"ls"}' },
    {
      type: 'block-end',
      index: 0,
      block: { type: 'tool-call', id: CallId('call-0'), name: 'bash', arguments: '{"cmd":"ls"}' },
    },
    { type: 'usage', usage: { inputTokens: 5, outputTokens: 4 } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ])
})

test('gemini stream: minted call ids are unique across turns, not just streams', () => {
  // Code Assist usually omits functionCall.id. Two separate responses that
  // each open one tool call at block 0 must still not collide: the ids
  // correlate results with calls by name on the NEXT request, so a repeat
  // would label an earlier result with a later call's function.
  const call = (): GeminiStreamEvent => ({
    response: {
      candidates: [{ content: { parts: [{ functionCall: { name: 'bash', args: {} } }] }, finishReason: 'STOP' }],
    },
  })
  const idOf = (chunks: StreamChunk[]): string => {
    const start = chunks.find(chunk => chunk.type === 'tool-call-delta')
    assert.ok(start?.type === 'tool-call-delta')
    return String(start.id)
  }
  const first = idOf(drain(new GeminiStreamTranslator(), [call()]))
  const second = idOf(drain(new GeminiStreamTranslator(), [call()]))
  assert.notEqual(first, second, 'two responses do not mint the same call id')
  // Two calls inside ONE stream must differ too.
  const both = drain(new GeminiStreamTranslator(), [call(), call()])
  const ids = both.filter(chunk => chunk.type === 'tool-call-delta').map(chunk => String(chunk.id))
  assert.equal(ids.length, 2)
  assert.notEqual(ids[0], ids[1])
})

test('gemini stream: thought signatures ride the finish chunk replay envelope', () => {
  const chunks = drain(new GeminiStreamTranslator(() => 'call-x'), [
    { response: { candidates: [{ content: { parts: [{ text: 'here goes' }] } }] } },
    {
      response: {
        candidates: [{
          content: { parts: [{ functionCall: { name: 'bash', args: {} }, thoughtSignature: 'sig-abc' }] },
          finishReason: 'STOP',
        }],
      },
    },
  ])
  const finish = chunks[chunks.length - 1]
  assert.ok(finish.type === 'finish')
  // One entry per emitted block, in stream order: the text block carries no
  // signature, the tool-call block carries the one the model issued.
  assert.deepEqual(finish.replayState, {
    response: { finishReason: 'STOP' },
    blocks: [{}, { thoughtSignature: 'sig-abc' }],
  })
})

test('gemini stream: a response with no signatures carries no replay envelope', () => {
  const chunks = drain(new GeminiStreamTranslator(), [
    { response: { candidates: [{ content: { parts: [{ text: 'hi' }] }, finishReason: 'STOP' }] } },
  ])
  const finish = chunks[chunks.length - 1]
  assert.ok(finish.type === 'finish')
  assert.equal(finish.replayState, undefined)
})

test('gemini stream: finish reasons map to harness reasons', () => {
  const maxed = drain(new GeminiStreamTranslator(), [
    { response: { candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason: 'MAX_TOKENS' }] } },
  ])
  assert.deepEqual(maxed[maxed.length - 1], { type: 'finish', reason: { kind: 'max-tokens' } })

  const unsafe = drain(new GeminiStreamTranslator(), [
    { response: { candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason: 'SAFETY' }] } },
  ])
  const finish = unsafe[unsafe.length - 1]
  assert.ok(finish.type === 'finish' && finish.reason.kind === 'error')
  if (finish.type === 'finish' && finish.reason.kind === 'error') {
    assert.match(finish.reason.failure.message, /SAFETY/)
  }
})

test('gemini stream: an empty response finishes as an error', () => {
  const chunks = drain(new GeminiStreamTranslator(), [])
  assert.deepEqual(chunks, [{
    type: 'finish',
    reason: {
      kind: 'error',
      failure: { message: 'model returned a completed response with no content', code: 'EMPTY_RESPONSE' },
    },
  }])
})

test('gemini stream: an in-band Google RPC error throws mapped', () => {
  const translator = new GeminiStreamTranslator()
  assert.throws(
    () => translator.push({ error: { code: 429, message: 'quota exhausted', status: 'RESOURCE_EXHAUSTED' } }),
    (error: unknown) => error instanceof LlmError && error.code === QUOTA_EXCEEDED_CODE,
  )
  assert.throws(
    () => new GeminiStreamTranslator().push({ error: { code: 401, message: 'bad token', status: 'UNAUTHENTICATED' } }),
    (error: unknown) => error instanceof LlmError && error.code === 'AUTH',
  )
})

test('gemini stream: a blocked prompt throws CONTENT_FILTER', () => {
  assert.throws(
    () => new GeminiStreamTranslator().push({ response: { promptFeedback: { blockReason: 'SAFETY' } } }),
    (error: unknown) => error instanceof LlmError
      && error.code === 'CONTENT_FILTER'
      && /SAFETY/.test(error.message),
  )
})

test('mapGeminiUsage keeps cached input disjoint and folds thoughts into output', () => {
  assert.deepEqual(mapGeminiUsage({ promptTokenCount: 10, candidatesTokenCount: 4 }), {
    inputTokens: 10,
    outputTokens: 4,
  })
  assert.deepEqual(mapGeminiUsage({}), { inputTokens: 0, outputTokens: 0 })
})

test('streamGemini consumes SSE bytes and finalizes at EOF', async () => {
  const payload = JSON.stringify({
    response: {
      candidates: [{ content: { parts: [{ text: 'hi' }] }, finishReason: 'STOP' }],
      usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 },
    },
  })
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(`data: ${payload}\r\n\r\n`))
      controller.close()
    },
  })
  const chunks: StreamChunk[] = []
  for await (const chunk of streamGemini(stream)) chunks.push(chunk)
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'hi' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'hi' } },
    { type: 'usage', usage: { inputTokens: 1, outputTokens: 1 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('streamGemini rejects a malformed SSE payload', async () => {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('data: {not json\n\n'))
      controller.close()
    },
  })
  await assert.rejects(async () => {
    for await (const chunk of streamGemini(stream)) void chunk
  }, /malformed SSE payload/)
})

// ---------------------------------------------------------------------------
// The adapter itself: catalog gating and the request envelope
// ---------------------------------------------------------------------------

const STATIC_GEMINI = [
  { id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro Preview', maxTokens: 4_096, contextWindow: 500_000 },
]

/** A TokenManager over an in-memory session; refresh never fires in these tests. */
function memoryTokens(initial: GeminiSession | undefined): TokenManager<GeminiSession> {
  let stored = initial
  return new TokenManager<GeminiSession>({
    displayName: 'Gemini (Subscription)',
    preemptMs: 0,
    load: () => Promise.resolve(stored),
    save: (session) => {
      stored = session
      return Promise.resolve()
    },
    remove: () => {
      stored = undefined
      return Promise.resolve()
    },
    refresh: session => Promise.resolve(session),
    isPermanent: () => false,
  })
}

/** Capture the single streaming request an adapter issues, answering an empty SSE body. */
function captureRequest(): { seen: SeenRequest[]; restore: () => void } {
  const real = globalThis.fetch
  const seen: SeenRequest[] = []
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    seen.push({
      url: input instanceof Request ? input.url : String(input),
      method: init?.method ?? 'GET',
      body: typeof init?.body === 'string' ? init.body : undefined,
    })
    return new Response(new ReadableStream<Uint8Array>({ start: controller => { controller.close() } }), {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    })
  }) as typeof fetch
  return { seen, restore: () => { globalThis.fetch = real } }
}

/** Drive one adapter stream to exhaustion and return the request it sent. */
async function streamOnce(
  adapter: GeminiAdapter,
  options: Partial<GenerateOptions> = {},
): Promise<Record<string, unknown>> {
  const capture = captureRequest()
  try {
    for await (const chunk of adapter.stream({
      provider: 'gemini',
      model: 'gemini-3-pro-preview',
      messages: [{
        id: MessageId('m1'),
        role: 'user',
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'user' },
      }],
      ...options,
    } as GenerateOptions)) void chunk
    assert.equal(capture.seen.length, 1, 'exactly one request was sent')
    assert.match(capture.seen[0].url, /v1internal:streamGenerateContent\?alt=sse$/)
    return JSON.parse(capture.seen[0].body ?? '{}') as Record<string, unknown>
  } finally {
    capture.restore()
  }
}

test('GeminiAdapter.listModels: logged out hides the provider, logged in serves the catalog', async () => {
  const loggedOut = new GeminiAdapter({
    models: STATIC_GEMINI, streamIdleTimeoutMs: 1_000, tokens: memoryTokens(undefined),
  })
  assert.deepEqual(await loggedOut.listModels('gemini'), [])

  const loggedIn = new GeminiAdapter({
    models: STATIC_GEMINI, streamIdleTimeoutMs: 1_000, tokens: memoryTokens(geminiSession),
  })
  assert.deepEqual(await loggedIn.listModels('gemini'), [{
    provider: 'gemini',
    id: 'gemini-3-pro-preview',
    name: 'Gemini 3 Pro Preview',
    inputModalities: ['text', 'image'],
  }])
})

test('GeminiAdapter.resolveModel: configured metadata wins over the defaults', async () => {
  const adapter = new GeminiAdapter({
    models: STATIC_GEMINI, streamIdleTimeoutMs: 1_000, tokens: memoryTokens(geminiSession),
  })
  assert.deepEqual(await adapter.resolveModel('gemini', 'gemini-3-pro-preview'), {
    provider: 'gemini',
    id: 'gemini-3-pro-preview',
    name: 'Gemini 3 Pro Preview',
    inputModalities: ['text', 'image'],
    context: { contextWindow: 500_000 },
    defaultMaxTokens: 4_096,
  })
  // An unknown id still resolves, on the provider-wide defaults.
  const unknown = await adapter.resolveModel('gemini', 'gemini-9')
  assert.equal(unknown.context?.contextWindow, 1_000_000)
  assert.equal(unknown.defaultMaxTokens, 65_536)
})

test('GeminiAdapter: the request carries the Code Assist envelope', async () => {
  const adapter = new GeminiAdapter({
    models: STATIC_GEMINI, streamIdleTimeoutMs: 1_000, tokens: memoryTokens(geminiSession),
  })
  const body = await streamOnce(adapter, { system: 'be brief' })
  assert.equal(body.model, 'gemini-3-pro-preview')
  assert.equal(body.project, 'managed-project-1', 'the resolved Code Assist project rides every request')
  assert.match(String(body.user_prompt_id), /^[0-9a-f-]{36}$/)
  const request = body.request as Record<string, unknown>
  assert.deepEqual(request.contents, [{ role: 'user', parts: [{ text: 'hi' }] }])
  assert.deepEqual(request.systemInstruction, { parts: [{ text: 'be brief' }] })
  const generationConfig = request.generationConfig as Record<string, unknown>
  // The configured per-model cap stands in when the caller names none.
  assert.equal(generationConfig.maxOutputTokens, 4_096)
  assert.deepEqual(generationConfig.thinkingConfig, { includeThoughts: true })
})

test('GeminiAdapter: thinkingConfig is withheld from models that do not think', async () => {
  const adapter = new GeminiAdapter({
    models: [{ id: 'gemini-2.5-flash-lite' }, { id: 'gemini-1.5-pro' }],
    streamIdleTimeoutMs: 1_000,
    tokens: memoryTokens(geminiSession),
  })
  for (const model of ['gemini-2.5-flash-lite', 'gemini-1.5-pro']) {
    const body = await streamOnce(adapter, { model })
    const generationConfig = (body.request as Record<string, unknown>).generationConfig as Record<string, unknown>
    assert.equal(generationConfig.thinkingConfig, undefined, `${model} gets no thinkingConfig`)
  }
  assert.equal(geminiThinks('gemini-3-pro-preview'), true)
  assert.equal(geminiThinks('gemini-2.5-flash-lite'), false)
})

test('GeminiAdapter: a system-only request fails before reaching the wire', async () => {
  const adapter = new GeminiAdapter({
    models: STATIC_GEMINI, streamIdleTimeoutMs: 1_000, tokens: memoryTokens(geminiSession),
  })
  const capture = captureRequest()
  try {
    await assert.rejects(async () => {
      for await (const chunk of adapter.stream({
        provider: 'gemini',
        model: 'gemini-3-pro-preview',
        system: 'only a system prompt',
        messages: [],
      } as unknown as GenerateOptions)) void chunk
    }, (error: unknown) => error instanceof LlmError && error.code === 'INVALID_REQUEST')
    assert.equal(capture.seen.length, 0, 'nothing was sent')
  } finally {
    capture.restore()
  }
})
