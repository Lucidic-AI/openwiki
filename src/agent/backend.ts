import { LocalShellBackend, type ReadResult } from "deepagents";

/**
 * OpenWiki backend that makes `read_file` results safe for text-only chat
 * providers.
 *
 * DeepAgents decides whether a file is text or binary purely from its
 * extension (`getMimeType` returns `application/octet-stream` for any
 * unrecognized extension, and its `extname` helper returns `""` when the only
 * dot is at index 0). As a result, common repository files with no usable
 * extension — `uv.lock`, `go.sum`, `Dockerfile`, `Makefile`, `LICENSE`,
 * `.gitignore`, `.env` — are treated as binary. The built-in `read_file` tool
 * then emits a `{ type: "file", data: <base64> }` content block.
 *
 * OpenAI-compatible endpoints (e.g. Azure OpenAI) only accept `text`,
 * `refusal`, `image_url`, and `input_audio` content parts, so that `file`
 * block is rejected with:
 *
 *   Error 400 Invalid value: 'file'. Supported values are: 'text',
 *   'refusal', 'image_url', and 'input_audio'.
 *
 * which aborts the whole run whenever the agent reads one of those files.
 *
 * DeepAgents' `read_file` tool emits `type: "image" | "audio" | "video" |
 * "file"` blocks for anything it thinks is binary. None of those type names are
 * in the OpenAI-compatible content schema — only `image_url` and `input_audio`
 * are — and `@langchain/openai` forwards the DeepAgents blocks verbatim (it
 * does not translate `image` to `image_url`), so reading a real image or audio
 * file triggers the identical HTTP 400. The only provider-safe content block
 * this backend lets through is `text`.
 *
 * This subclass overrides {@link read} to inspect the actual bytes DeepAgents
 * returned. Files that are really UTF-8 text are returned as `text/plain` so
 * the tool emits a normal `text` block; every non-text file (image, audio,
 * video, PDF, archive, …) is replaced with a short `text` placeholder so no
 * non-text content block is ever produced. OpenWiki documents code, so summar-
 * izing binary files rather than shipping their raw bytes is also the right
 * behavior. The single backend instance is shared by the main agent and every
 * subagent, so this fixes all read paths at once.
 */
export class OpenWikiLocalShellBackend extends LocalShellBackend {
  override async read(
    filePath: string,
    offset?: number,
    limit?: number,
  ): Promise<ReadResult> {
    // Mirror the defaults DeepAgents' FilesystemBackend.read applies so the
    // slicing below lines up with what the tool expects.
    const resolvedOffset = offset ?? DEFAULT_READ_OFFSET;
    const resolvedLimit = limit ?? DEFAULT_READ_LIMIT;
    const result = await super.read(filePath, resolvedOffset, resolvedLimit);

    return normalizeReadResult(result, filePath, resolvedOffset, resolvedLimit);
  }
}

const DEFAULT_READ_OFFSET = 0;
// Matches FilesystemBackend.read's default `limit` in deepagents.
const DEFAULT_READ_LIMIT = 500;
const TEXT_MIME_TYPE = "text/plain";

/**
 * Same 10MB ceiling DeepAgents' `read_file` tool enforces on binary reads
 * (`MAX_BINARY_READ_SIZE_BYTES`). DeepAgents only applies it on the binary
 * branch, so when this backend reclassifies a large binary-typed file as text
 * the guard would be skipped; re-applying it here keeps an oversized file a
 * graceful "too large" message instead of a request the provider rejects.
 */
const MAX_READ_BYTES = 10 * 1024 * 1024;

/**
 * Byte value that never appears in well-formed UTF-8 text and reliably marks
 * binary formats (images, archives, compiled objects). This is the same
 * heuristic Git uses to classify a blob as binary.
 */
const NUL_BYTE = 0;

/**
 * DeepAgents' sentinel for an existing-but-empty file. Reusing the exact
 * string keeps reclassified empty files indistinguishable from files
 * DeepAgents already recognized as text.
 */
const EMPTY_CONTENT_WARNING =
  "System reminder: File exists but has empty contents";

/**
 * Reshape a {@link ReadResult} so that a text-only provider will accept it.
 *
 * Exported for unit testing; {@link OpenWikiLocalShellBackend.read} is the only
 * production caller.
 *
 * @param result - Result returned by `LocalShellBackend.read`
 * @param filePath - Path that was read (used only for the placeholder message)
 * @param offset - Resolved line offset the read was requested with
 * @param limit - Resolved line limit the read was requested with
 */
export function normalizeReadResult(
  result: ReadResult,
  filePath: string,
  offset: number,
  limit: number,
): ReadResult {
  // Errors and text content are already provider-safe; pass them through
  // untouched so behavior for recognized text files is unchanged.
  if (result.error !== undefined) {
    return result;
  }

  const { content } = result;

  if (typeof content === "string" || !(content instanceof Uint8Array)) {
    return result;
  }

  const mimeType = result.mimeType ?? "application/octet-stream";

  // Re-apply DeepAgents' binary size guard before decoding: a >10MB file that
  // is NUL-free but has few newlines would otherwise be returned whole as one
  // enormous text block that overflows the provider's request limit.
  if (content.byteLength > MAX_READ_BYTES) {
    return {
      content: `[file '${filePath}' omitted: ${content.byteLength} bytes exceeds the ${MAX_READ_BYTES}-byte read limit]`,
      mimeType: TEXT_MIME_TYPE,
    };
  }

  // Files flagged as binary only because of an unrecognized extension are
  // almost always UTF-8 text. Detect that from the bytes and return them as
  // text so the model can actually read them.
  if (isProbablyTextBytes(content)) {
    return sliceDecodedText(content, offset, limit);
  }

  // Any real binary (image, audio, video, PDF, archive, …). DeepAgents would
  // emit a non-text content block here — image/audio/video/file — none of
  // which an OpenAI-compatible provider accepts, so summarize it as text
  // instead. The model still learns the file exists, its type, and its size.
  return {
    content: `[binary file '${filePath}' omitted: ${content.byteLength} bytes, type ${mimeType}]`,
    mimeType: TEXT_MIME_TYPE,
  };
}

/**
 * Heuristic byte-level text detection: a file is treated as text when it
 * contains no NUL byte. An empty file is trivially text.
 */
export function isProbablyTextBytes(bytes: Uint8Array): boolean {
  return !bytes.includes(NUL_BYTE);
}

/**
 * Decode text bytes and apply the same empty-file and offset/limit slicing that
 * DeepAgents' `FilesystemBackend.read` text path performs, so a reclassified
 * file reads identically to a natively recognized text file.
 */
function sliceDecodedText(
  bytes: Uint8Array,
  offset: number,
  limit: number,
): ReadResult {
  const text = new TextDecoder("utf-8", { fatal: false }).decode(bytes);

  if (text.trim() === "") {
    return { content: EMPTY_CONTENT_WARNING, mimeType: TEXT_MIME_TYPE };
  }

  const lines = text.split("\n");

  if (offset >= lines.length) {
    return {
      error: `Line offset ${offset} exceeds file length (${lines.length} lines)`,
    };
  }

  const endIdx = Math.min(offset + limit, lines.length);

  return {
    content: lines.slice(offset, endIdx).join("\n"),
    mimeType: TEXT_MIME_TYPE,
  };
}
