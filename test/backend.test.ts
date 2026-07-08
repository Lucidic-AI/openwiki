import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  isProbablyTextBytes,
  normalizeReadResult,
  OpenWikiLocalShellBackend,
} from "../src/agent/backend.ts";

const EMPTY_CONTENT_WARNING =
  "System reminder: File exists but has empty contents";

function octetStream(text: string): {
  content: Uint8Array;
  mimeType: string;
} {
  return {
    content: new TextEncoder().encode(text),
    mimeType: "application/octet-stream",
  };
}

describe("isProbablyTextBytes", () => {
  test("treats NUL-free bytes as text", () => {
    expect(isProbablyTextBytes(new TextEncoder().encode("FROM node:20"))).toBe(
      true,
    );
  });

  test("treats an empty buffer as text", () => {
    expect(isProbablyTextBytes(new Uint8Array(0))).toBe(true);
  });

  test("treats bytes containing a NUL as binary", () => {
    expect(isProbablyTextBytes(new Uint8Array([0x89, 0x50, 0x00, 0x4e]))).toBe(
      false,
    );
  });
});

describe("normalizeReadResult", () => {
  test("reclassifies text bytes with an unknown extension as text/plain", () => {
    // This is the exact shape LocalShellBackend.read returns for uv.lock,
    // Dockerfile, etc. A text/* mimeType steers the read_file tool to its
    // text branch, so it emits a `text` block instead of the unsupported
    // `file` block that Azure/OpenAI-compatible endpoints reject.
    const result = normalizeReadResult(
      octetStream("version = 1\nrequires-python = '>=3.12'"),
      "/uv.lock",
      0,
      500,
    );

    expect(result.error).toBeUndefined();
    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toBe("version = 1\nrequires-python = '>=3.12'");
  });

  test("replaces genuinely binary content with a text placeholder", () => {
    const bytes = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0x00, 0x01]);
    const result = normalizeReadResult(
      { content: bytes, mimeType: "application/octet-stream" },
      "/archive.unknown",
      0,
      500,
    );

    expect(result.error).toBeUndefined();
    expect(result.mimeType).toBe("text/plain");
    expect(typeof result.content).toBe("string");
    expect(result.content).toContain("binary file '/archive.unknown' omitted");
    expect(result.content).toContain("6 bytes");
  });

  // DeepAgents emits `{ type: "image" }` (not `image_url`), which
  // OpenAI-compatible providers reject with the same HTTP 400 as `file`, so
  // images must be downgraded to text rather than passed through.
  test("downgrades images to a text placeholder", () => {
    const bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x00]);
    const result = normalizeReadResult(
      { content: bytes, mimeType: "image/png" },
      "/logo.png",
      0,
      500,
    );

    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toContain("binary file '/logo.png' omitted");
    expect(result.content).toContain("image/png");
  });

  test("downgrades audio to a text placeholder", () => {
    const bytes = new Uint8Array([0x49, 0x44, 0x33, 0x00]);
    const result = normalizeReadResult(
      { content: bytes, mimeType: "audio/mpeg" },
      "/clip.mp3",
      0,
      500,
    );

    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toContain("binary file '/clip.mp3' omitted");
  });

  test("downgrades video to a text placeholder", () => {
    const bytes = new Uint8Array([0x00, 0x00, 0x00, 0x18]);
    const result = normalizeReadResult(
      { content: bytes, mimeType: "video/mp4" },
      "/demo.mp4",
      0,
      500,
    );

    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toContain("binary file '/demo.mp4' omitted");
  });

  test("caps an oversized text-like file instead of decoding it whole", () => {
    // >10MB, NUL-free, single line: would otherwise be returned as one giant
    // text block that overflows the provider request.
    const huge = new Uint8Array(10 * 1024 * 1024 + 1).fill(0x41);
    const result = normalizeReadResult(
      { content: huge, mimeType: "application/octet-stream" },
      "/bundle.b64",
      0,
      500,
    );

    expect(result.error).toBeUndefined();
    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toContain("exceeds the");
    expect(result.content).toContain("read limit");
  });

  test("passes through results that are already text", () => {
    const result = normalizeReadResult(
      { content: "already text", mimeType: "text/plain" },
      "/README.md",
      0,
      500,
    );

    expect(result.content).toBe("already text");
    expect(result.mimeType).toBe("text/plain");
  });

  test("passes through error results", () => {
    const result = normalizeReadResult(
      { error: "File '/missing' not found" },
      "/missing",
      0,
      500,
    );

    expect(result.error).toBe("File '/missing' not found");
    expect(result.content).toBeUndefined();
  });

  test("returns the empty-content warning for an empty text-like file", () => {
    const result = normalizeReadResult(octetStream(""), "/empty.lock", 0, 500);

    expect(result.content).toBe(EMPTY_CONTENT_WARNING);
    expect(result.mimeType).toBe("text/plain");
  });

  test("applies offset/limit slicing like the text read path", () => {
    const result = normalizeReadResult(
      octetStream("l0\nl1\nl2\nl3\nl4"),
      "/lines.lock",
      1,
      2,
    );

    expect(result.content).toBe("l1\nl2");
  });

  test("reports an offset past end-of-file", () => {
    const result = normalizeReadResult(
      octetStream("l0\nl1"),
      "/lines.lock",
      9,
      2,
    );

    expect(result.error).toContain("Line offset 9 exceeds file length");
  });
});

describe("OpenWikiLocalShellBackend.read (integration)", () => {
  let root: string;
  let backend: OpenWikiLocalShellBackend;

  beforeAll(async () => {
    root = await mkdtemp(path.join(tmpdir(), "openwiki-backend-"));
    backend = new OpenWikiLocalShellBackend({
      rootDir: root,
      virtualMode: true,
    });

    await writeFile(
      path.join(root, "Dockerfile"),
      "FROM node:20\nRUN npm ci\n",
    );
    await writeFile(
      path.join(root, "uv.lock"),
      "version = 1\nrequires-python = '>=3.12'\n",
    );
    await writeFile(path.join(root, ".gitignore"), "node_modules\ndist\n");
    await writeFile(path.join(root, "LICENSE"), "MIT License\n");
    await writeFile(path.join(root, "README.md"), "# Title\n\nBody\n");
    await writeFile(path.join(root, "empty.cfg"), "");
    await writeFile(
      path.join(root, "data.bin"),
      Buffer.from([0x00, 0x01, 0x02, 0xff, 0xfe, 0x00]),
    );
    // Minimal PNG signature so the extension-based mimeType is image/png.
    await writeFile(
      path.join(root, "logo.png"),
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]),
    );
  });

  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  // These files are exactly the ones that used to produce a `file` content
  // block and crash the run with HTTP 400 on OpenAI-compatible providers.
  test.each([
    ["/Dockerfile", "FROM node:20"],
    ["/uv.lock", "version = 1"],
    ["/.gitignore", "node_modules"],
    ["/LICENSE", "MIT License"],
  ])("reads %s as text", async (virtualPath, expectedSubstring) => {
    const result = await backend.read(virtualPath);

    expect(result.error).toBeUndefined();
    expect(typeof result.content).toBe("string");
    expect(result.mimeType?.startsWith("text/")).toBe(true);
    expect(result.content).toContain(expectedSubstring);
  });

  test("still reads a recognized text file normally", async () => {
    const result = await backend.read("/README.md");

    expect(typeof result.content).toBe("string");
    expect(result.content).toContain("# Title");
  });

  test("returns a text placeholder for a real binary file", async () => {
    const result = await backend.read("/data.bin");

    expect(result.error).toBeUndefined();
    expect(typeof result.content).toBe("string");
    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toContain("binary file '/data.bin' omitted");
  });

  test("downgrades a real image to a text placeholder", async () => {
    const result = await backend.read("/logo.png");

    expect(typeof result.content).toBe("string");
    expect(result.mimeType).toBe("text/plain");
    expect(result.content).toContain("binary file '/logo.png' omitted");
  });
});
