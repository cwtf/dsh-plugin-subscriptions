/**
 * Translate between the harness message vocabulary and the Gemini generateContent
 * wire format used by the gemini provider's Cloud Code Assist endpoint: request
 * content assembly, tool schema mapping, and a push-model SSE-event →
 * StreamChunk state machine ({@link GeminiStreamTranslator}) so tests need no
 * streams.
 *
 * Code Assist wraps every response chunk in an envelope
 * (`{ response: GenerateContentResponse, traceId?, ... }`); the translator
 * reads the inner Vertex-flavored response only.
 */

import { randomUUID } from 'node:crypto'
import {
  CallId,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  EMPTY_RESPONSE_CODE,
  isContextWindowExceededError,
  isQuotaExceededError,
  LlmError,
  QUOTA_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  ReplayEnvelope,
  StreamChunk,
  TokenUsage,
  ToolResultBlock,
  ToolSchema,
} from '@deepseek-ai/dsh-llm'
import { parseSse } from './sse.js'
import type { TranslatableMessage } from './resolved.js'

/** One Gemini request content entry (`user` or `model` role). */
export interface GeminiContent {
  role: 'user' | 'model'
  parts: Record<string, unknown>[]
}

/** Flatten a tool result's content to plain text for `functionResponse`. */
function toolResultText(block: ToolResultBlock): string {
  return block.content.map(part => (part.type === 'text' ? part.text : '')).join('')
}

/**
 * Per-block replay metadata this translator stores in the harness
 * {@link ReplayEnvelope}, one entry per emitted block in stream order.
 */
export interface GeminiBlockReplay {
  /**
   * Gemini 3's opaque thought signature for the part that produced the block.
   * It must be echoed verbatim on the next request: the API validates
   * signatures on replayed function calls and rejects a call that arrives
   * without the one it was issued with.
   */
  thoughtSignature?: string
}

/**
 * This translator's per-block replay entries for one assistant message, when
 * the harness handed back an envelope that still lines up with the message's
 * blocks. Assembly drops envelope entries alongside the blocks they describe,
 * so a length mismatch means the envelope no longer describes this content and
 * is ignored rather than misapplied.
 */
function replayBlocks(message: TranslatableMessage): readonly unknown[] | undefined {
  const source = message.source
  if (source === undefined || source.kind !== 'model') return undefined
  const envelope: unknown = source.replayState
  if (!isPlainObject(envelope)) return undefined
  const blocks: unknown = envelope.blocks
  if (!Array.isArray(blocks) || blocks.length !== message.content.length) return undefined
  return blocks
}

/** The thought signature recorded for one block position, when there is one. */
function thoughtSignatureAt(blocks: readonly unknown[] | undefined, index: number): string | undefined {
  const entry: unknown = blocks?.[index]
  if (!isPlainObject(entry)) return undefined
  const signature: unknown = entry.thoughtSignature
  return typeof signature === 'string' && signature.length > 0 ? signature : undefined
}

/** Parse a tool call's raw JSON arguments into Gemini's object-shaped `args`. */
function parseToolArgs(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
    return {}
  } catch {
    // The model produced malformed JSON; an empty object keeps the request valid.
    return {}
  }
}

/**
 * Convert harness messages into Gemini `contents`. Consecutive same-role
 * messages merge into one content with multiple parts; system-role messages
 * are handled by {@link toGeminiSystem} and skipped here. Reasoning blocks
 * are not replayed (v1). A tool result becomes a `functionResponse` part
 * carrying the call's id AND name (resolved from the matching earlier
 * tool-call block: the result itself names only the id, and the wire format
 * correlates by name when the model issued no id). Text and functionCall
 * parts carry back the `thoughtSignature` the model issued them with, read
 * from the assistant message's replay envelope. Images must arrive
 * pre-resolved ({@link TranslatableMessage}); an unresolved ImageBlock is
 * skipped because its bytes are unreachable here.
 * @param messages - ordered conversation messages with resolved images.
 * @returns Gemini contents in conversation order.
 */
export function toGeminiContents(messages: readonly TranslatableMessage[]): GeminiContent[] {
  // First pass: tool call id → name, so results can name their function. Ids
  // are unique per call (the translator mints one when the model omits it), so
  // a later turn cannot overwrite an earlier call's name.
  const callNames = new Map<string, string>()
  for (const message of messages) {
    for (const block of message.content) {
      if (block.type === 'tool-call') callNames.set(String(block.id), block.name)
    }
  }
  const out: GeminiContent[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    const role = message.role === 'assistant' ? 'model' : 'user'
    const replay = replayBlocks(message)
    const parts: Record<string, unknown>[] = []
    for (const [index, block] of message.content.entries()) {
      const thoughtSignature = thoughtSignatureAt(replay, index)
      const signed = thoughtSignature === undefined ? {} : { thoughtSignature }
      switch (block.type) {
        case 'text':
          parts.push({ text: block.text, ...signed })
          break
        case 'tool-call':
          parts.push({
            functionCall: {
              id: String(block.id),
              name: block.name,
              args: parseToolArgs(block.arguments),
            },
            ...signed,
          })
          break
        case 'tool-result': {
          const id = String(block.toolCallId)
          const text = toolResultText(block)
          parts.push({
            functionResponse: {
              id,
              name: callNames.get(id) ?? id,
              // Gemini distinguishes a failed call from a successful one by
              // which key the response object carries.
              response: block.isError === true ? { error: text } : { output: text },
            },
          })
          break
        }
        case 'image':
          if ('dataBase64' in block) {
            parts.push({ inlineData: { mimeType: block.mediaType, data: block.dataBase64 } })
          }
          // An unresolved ImageBlock carries only an attachment reference; the
          // adapter resolves images before translation, so this is skipped.
          break
        default:
          // reasoning (not replayed), unknown blocks.
          break
      }
    }
    if (parts.length === 0) continue
    const last = out[out.length - 1]
    if (last !== undefined && last.role === role) last.parts.push(...parts)
    else out.push({ role, parts })
  }
  return out
}

/**
 * Build the Gemini `systemInstruction` content from the explicit system
 * prompt plus any system-role messages, or undefined when there is none.
 * @param system - explicit system prompt, which takes precedence position.
 * @param messages - conversation messages; their system-role text is appended.
 * @returns the system instruction content, or undefined.
 */
export function toGeminiSystem(
  system?: string,
  messages?: readonly TranslatableMessage[],
): { parts: { text: string }[] } | undefined {
  const texts: string[] = []
  if (system !== undefined && system.length > 0) texts.push(system)
  for (const message of messages ?? []) {
    if (message.role !== 'system') continue
    for (const block of message.content) {
      if (block.type === 'text') texts.push(block.text)
    }
  }
  return texts.length > 0 ? { parts: texts.map(text => ({ text })) } : undefined
}

/**
 * Keywords Gemini's function-declaration schema accepts. Its `parameters` is
 * an OpenAPI 3.0 Schema subset, NOT full JSON Schema: an unknown key fails the
 * whole request with `400 INVALID_ARGUMENT` ("Unknown name ..."), so the
 * harness's plain JSON Schema has to be filtered down rather than passed
 * through. `additionalProperties` and `$schema` are the two that bite in
 * practice — this plugin's own tools declare `additionalProperties: false`.
 */
const GEMINI_SCHEMA_KEYS: readonly string[] = [
  'type', 'format', 'title', 'description', 'nullable', 'enum', 'default', 'example',
  'items', 'minItems', 'maxItems',
  'properties', 'required', 'minProperties', 'maxProperties', 'propertyOrdering',
  'minLength', 'maxLength', 'pattern',
  'minimum', 'maximum',
  'anyOf',
]

/** `format` values Gemini recognizes, by type; anything else is dropped. */
const GEMINI_FORMATS: Readonly<Record<string, readonly string[]>> = {
  string: ['enum', 'date-time'],
  number: ['float', 'double'],
  integer: ['int32', 'int64'],
}

/** True for a plain (non-array, non-null) object, the shape a subschema takes. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Reduce one JSON Schema to the OpenAPI subset Gemini accepts: unknown
 * keywords are dropped, `const` folds into a single-value `enum`, a union
 * `type` (e.g. `['string', 'null']`) collapses to its first non-null member
 * plus `nullable`, and `oneOf`/`allOf` become `anyOf` (Gemini's only
 * combinator). Recurses through `properties`, `items`, and `anyOf`.
 * @param schema - the tool's JSON Schema fragment.
 * @returns an equivalent fragment Gemini will accept.
 */
export function sanitizeGeminiSchema(schema: unknown): Record<string, unknown> {
  if (!isPlainObject(schema)) return {}
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(schema)) {
    if (!GEMINI_SCHEMA_KEYS.includes(key)) continue
    switch (key) {
      case 'properties': {
        if (!isPlainObject(value)) break
        const properties: Record<string, unknown> = {}
        for (const [name, sub] of Object.entries(value)) properties[name] = sanitizeGeminiSchema(sub)
        out.properties = properties
        break
      }
      case 'items':
        out.items = sanitizeGeminiSchema(value)
        break
      case 'anyOf':
        if (Array.isArray(value)) out.anyOf = value.map(entry => sanitizeGeminiSchema(entry))
        break
      default:
        out[key] = value
        break
    }
  }
  // `oneOf`/`allOf` carry real constraints, so they are remapped rather than
  // dropped: Gemini models both as `anyOf`.
  for (const combinator of ['oneOf', 'allOf'] as const) {
    const value = schema[combinator]
    if (out.anyOf === undefined && Array.isArray(value)) {
      out.anyOf = value.map(entry => sanitizeGeminiSchema(entry))
    }
  }
  // `const` is JSON Schema's single-value constraint; Gemini spells it `enum`.
  if (out.enum === undefined && schema.const !== undefined) out.enum = [schema.const]
  // A union type is JSON Schema only: Gemini takes one type plus `nullable`.
  if (Array.isArray(out.type)) {
    const members = out.type.filter((entry): entry is string => typeof entry === 'string')
    const concrete = members.find(entry => entry !== 'null')
    if (concrete === undefined) delete out.type
    else out.type = concrete
    if (members.includes('null')) out.nullable = true
  }
  // An unrecognized `format` is rejected the same way an unknown key is.
  if (typeof out.format === 'string' && typeof out.type === 'string'
    && !(GEMINI_FORMATS[out.type] ?? []).includes(out.format)) {
    delete out.format
  }
  return out
}

/**
 * Map harness tool schemas to one Gemini `tools` entry, with each schema
 * reduced to the subset Gemini accepts ({@link sanitizeGeminiSchema}).
 * @param tools - tool schemas from the request.
 * @returns a single-entry `tools` array holding every function declaration.
 */
export function toGeminiTools(tools: readonly ToolSchema[]): Record<string, unknown>[] {
  return [{
    functionDeclarations: tools.map(tool => ({
      name: tool.name,
      description: tool.description,
      parameters: sanitizeGeminiSchema(tool.parameters),
    })),
  }]
}

/** One candidate part of a Gemini response. */
interface GeminiPart {
  text?: string
  /** True on thought (reasoning summary) parts. */
  thought?: boolean
  /** Gemini 3's opaque signature for this part; replayed verbatim next turn. */
  thoughtSignature?: string
  functionCall?: { id?: string; name?: string; args?: Record<string, unknown> }
}

/** Gemini usage metadata; counts are cumulative on the final chunk. */
export interface GeminiUsageMetadata {
  promptTokenCount?: number
  candidatesTokenCount?: number
  cachedContentTokenCount?: number
  thoughtsTokenCount?: number
}

/** The inner Vertex-flavored response of one Code Assist chunk. */
interface GeminiResponse {
  candidates?: {
    content?: { parts?: GeminiPart[] }
    finishReason?: string
  }[]
  promptFeedback?: { blockReason?: string }
  usageMetadata?: GeminiUsageMetadata
}

/** One parsed Code Assist SSE event: the envelope around the response. */
export interface GeminiStreamEvent {
  response?: GeminiResponse
  traceId?: string
  /** Google RPC error payload (delivered in-band on HTTP 200 streams). */
  error?: { code?: number; message?: string; status?: string }
}

/**
 * Map Gemini usage metadata to disjoint harness counts: cached input is
 * subtracted out of `inputTokens`, and thought tokens ride `reasoningTokens`
 * inside `outputTokens` (mirroring the Responses translator's shape).
 * @param usage - wire usage from the final chunk.
 * @returns harness token usage.
 */
export function mapGeminiUsage(usage: GeminiUsageMetadata): TokenUsage {
  const cached = usage.cachedContentTokenCount
  const thoughts = usage.thoughtsTokenCount
  return {
    inputTokens: (usage.promptTokenCount ?? 0) - (cached ?? 0),
    outputTokens: (usage.candidatesTokenCount ?? 0) + (thoughts ?? 0),
    ...cached !== undefined ? { cacheReadTokens: cached } : {},
    ...thoughts !== undefined ? { reasoningTokens: thoughts } : {},
  }
}

/**
 * Classify an in-band Google RPC error payload into a thrown LlmError.
 * @param error - the wire error object.
 * @returns the mapped error (quota, context overflow, auth, otherwise SERVER).
 */
export function geminiFailure(error: GeminiStreamEvent['error']): LlmError {
  const status = error?.status ?? 'UNKNOWN'
  const message = error?.message ?? `Gemini reported ${status}`
  if (status === 'RESOURCE_EXHAUSTED' || isQuotaExceededError(message)) {
    return new LlmError(message, QUOTA_EXCEEDED_CODE)
  }
  if (status === 'UNAUTHENTICATED' || status === 'PERMISSION_DENIED') return new LlmError(message, 'AUTH')
  if (status === 'INVALID_ARGUMENT' && isContextWindowExceededError(message)) {
    return new LlmError(message, CONTEXT_WINDOW_EXCEEDED_CODE)
  }
  return new LlmError(message, 'SERVER')
}

/** One open harness block under assembly. */
interface OpenBlock {
  index: number
  kind: 'text' | 'reasoning'
  text: string
}

/**
 * Push-model Gemini SSE translator: feed each parsed Code Assist envelope to
 * {@link push} and collect the emitted harness StreamChunks. Text and thought
 * parts arrive as deltas and extend the currently-open block of their kind;
 * function calls arrive atomically and open+close a tool-call block at once.
 * `usage` is emitted before the terminal `finish`, and nothing is emitted
 * after it. In-band `error` payloads and a `promptFeedback` block throw
 * {@link LlmError}.
 *
 * Per-block `thoughtSignature`s are collected as they arrive and ride the
 * terminal `finish` chunk's {@link ReplayEnvelope}, which
 * {@link toGeminiContents} reads back on the next request.
 */
export class GeminiStreamTranslator {
  private openBlock: OpenBlock | undefined
  private nextIndex = 0
  private sawAnyBlock = false
  private sawToolCall = false
  private usage: GeminiUsageMetadata | undefined
  private finishReason: string | undefined
  /** Replay metadata per emitted block, indexed by the block's own index. */
  private blockReplay: GeminiBlockReplay[] = []
  /** Set once {@link finish} produced the terminal finish chunk. */
  terminated = false

  /**
   * @param newCallId - mints an id for a functionCall the model left
   *   unidentified. Must be unique across the whole conversation, not just
   *   this stream: ids correlate tool results with their call by name on the
   *   next request, so a value that repeats across turns mislabels them.
   *   Injectable so tests can assert exact chunks.
   */
  constructor(private readonly newCallId: () => string = () => `call-${randomUUID()}`) {}

  /** Record the signature the model issued for one emitted block, when any. */
  private sign(index: number, part: GeminiPart): void {
    if (typeof part.thoughtSignature !== 'string' || part.thoughtSignature.length === 0) return
    this.blockReplay[index] = { thoughtSignature: part.thoughtSignature }
  }

  /** Append text to the open block of `kind`, closing any open block of the other kind. */
  private delta(kind: OpenBlock['kind'], text: string, chunks: StreamChunk[]): void {
    if (this.openBlock?.kind !== kind) {
      this.closeOpen(chunks)
      this.openBlock = { index: this.nextIndex++, kind, text: '' }
      chunks.push({ type: 'block-start', index: this.openBlock.index, blockType: kind })
    }
    this.sawAnyBlock = true
    this.openBlock.text += text
    chunks.push({ type: kind === 'text' ? 'text-delta' : 'reasoning-delta', index: this.openBlock.index, text })
  }

  private closeOpen(chunks: StreamChunk[]): void {
    const block = this.openBlock
    if (block === undefined) return
    this.openBlock = undefined
    const closed: ContentBlock = block.kind === 'text'
      ? { type: 'text', text: block.text }
      : { type: 'reasoning', text: block.text }
    chunks.push({ type: 'block-end', index: block.index, block: closed })
  }

  /**
   * Process one parsed Code Assist SSE envelope.
   * @param event - the parsed event object.
   * @returns the StreamChunks this event produced (possibly none).
   */
  push(event: GeminiStreamEvent): StreamChunk[] {
    if (this.terminated) return []
    if (event.error !== undefined) throw geminiFailure(event.error)
    const chunks: StreamChunk[] = []
    const response = event.response
    if (response === undefined) return chunks
    if (response.promptFeedback?.blockReason !== undefined) {
      throw new LlmError(
        `gemini blocked the prompt (${response.promptFeedback.blockReason})`,
        'CONTENT_FILTER',
      )
    }
    if (response.usageMetadata !== undefined) this.usage = response.usageMetadata
    const candidate = response.candidates?.[0]
    if (candidate === undefined) return chunks
    for (const part of candidate.content?.parts ?? []) {
      if (part.functionCall !== undefined) {
        this.closeOpen(chunks)
        this.sawAnyBlock = true
        this.sawToolCall = true
        // Code Assist usually omits the id; a minted one must be unique across
        // turns, not merely within this stream.
        const id = part.functionCall.id ?? this.newCallId()
        const name = part.functionCall.name ?? ''
        const args = JSON.stringify(part.functionCall.args ?? {})
        const index = this.nextIndex++
        this.sign(index, part)
        chunks.push({ type: 'block-start', index, blockType: 'tool-call' })
        chunks.push({ type: 'tool-call-delta', index, id: CallId(id), name, argumentsDelta: args })
        chunks.push({
          type: 'block-end',
          index,
          block: { type: 'tool-call', id: CallId(id), name, arguments: args },
        })
      } else if (part.thought === true) {
        if (typeof part.text === 'string' && part.text.length > 0) {
          this.delta('reasoning', part.text, chunks)
          this.sign(this.openBlock?.index ?? 0, part)
        }
      } else if (typeof part.text === 'string' && part.text.length > 0) {
        this.delta('text', part.text, chunks)
        this.sign(this.openBlock?.index ?? 0, part)
      }
    }
    if (candidate.finishReason !== undefined) this.finishReason = candidate.finishReason
    return chunks
  }

  /**
   * Produce the terminal chunks at end of stream. Code Assist's SSE stream
   * simply ends after the final chunk (no `message_stop`/`[DONE]` sentinel),
   * so the byte stream's EOF is the completion signal.
   * @returns the closing block-end (when open), usage, and finish chunks.
   */
  finish(): StreamChunk[] {
    if (this.terminated) return []
    this.terminated = true
    const chunks: StreamChunk[] = []
    this.closeOpen(chunks)
    if (this.usage !== undefined) chunks.push({ type: 'usage', usage: mapGeminiUsage(this.usage) })
    if (!this.sawAnyBlock) {
      chunks.push({
        type: 'finish',
        reason: {
          kind: 'error',
          failure: {
            message: this.finishReason !== undefined && this.finishReason !== 'STOP'
              ? `model returned no content (finishReason ${this.finishReason})`
              : 'model returned a completed response with no content',
            code: EMPTY_RESPONSE_CODE,
          },
        },
      })
      return chunks
    }
    const replayState = this.replayEnvelope()
    const carry = replayState === undefined ? {} : { replayState }
    switch (this.finishReason) {
      case 'MAX_TOKENS':
        chunks.push({ type: 'finish', reason: { kind: 'max-tokens' }, ...carry })
        break
      case 'SAFETY':
      case 'RECITATION':
      case 'BLOCKLIST':
      case 'PROHIBITED_CONTENT':
      case 'SPII':
      case 'MALFORMED_FUNCTION_CALL':
        chunks.push({
          type: 'finish',
          reason: {
            kind: 'error',
            failure: { message: `gemini stopped generation (${this.finishReason})`, code: 'CONTENT_FILTER' },
          },
        })
        break
      default:
        // STOP, FINISH_REASON_UNSPECIFIED, or an absent reason on a content-bearing stream.
        chunks.push({
          type: 'finish',
          reason: { kind: this.sawToolCall ? 'tool-calls' : 'stop' },
          ...carry,
        })
        break
    }
    return chunks
  }

  /**
   * The replay envelope for this response, or undefined when the model issued
   * no signatures (every entry empty). `blocks` carries exactly one entry per
   * emitted block, in stream order, because assembly discards an envelope
   * whose length stops matching the blocks it kept.
   */
  private replayEnvelope(): ReplayEnvelope | undefined {
    if (!this.blockReplay.some(entry => entry?.thoughtSignature !== undefined)) return undefined
    const blocks: GeminiBlockReplay[] = []
    for (let index = 0; index < this.nextIndex; index++) blocks.push(this.blockReplay[index] ?? {})
    return {
      response: { ...this.finishReason === undefined ? {} : { finishReason: this.finishReason } },
      blocks,
    }
  }
}

/**
 * Consume a Code Assist SSE byte stream and yield harness StreamChunks. The
 * stream ends without a sentinel, so EOF finalizes the response.
 * @param stream - raw response body.
 * @param onActivity - transport-activity callback for the idle watchdog.
 * @returns the chunk stream.
 */
export async function* streamGemini(
  stream: ReadableStream<Uint8Array>,
  onActivity?: () => void,
): AsyncGenerator<StreamChunk> {
  const translator = new GeminiStreamTranslator()
  for await (const sseEvent of parseSse(stream, onActivity)) {
    let event: GeminiStreamEvent
    try {
      event = JSON.parse(sseEvent.data) as GeminiStreamEvent
    } catch {
      throw new LlmError(`malformed SSE payload: ${sseEvent.data.slice(0, 120)}`, 'MALFORMED_RESPONSE')
    }
    yield* translator.push(event)
  }
  yield* translator.finish()
}
