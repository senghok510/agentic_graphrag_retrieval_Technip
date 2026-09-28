// Client for the FastAPI pipeline's SSE stream (see chatbot/src/pipeline/streaming.py
// and chatbot/src/pipeline/trace.py for the exact event shapes this mirrors).

export interface ChunkItem {
  file_name: string;
  page_number: string;
  source?: string;
  score?: number;
  text?: string;
}

export interface RelationItem {
  subject: string;
  label: string;
  object: string;
  description?: string;
  score?: number;
  channel?: string;
}

export interface EntityItem {
  name: string;
  category?: string;
  score?: number;
}

export type PayloadBlock = {
  title: string;
} & (
  | { kind: "chunks"; items: ChunkItem[] }
  | { kind: "relations"; items: RelationItem[] }
  | { kind: "entities"; items: EntityItem[] }
  | { kind: "text"; items: string }
  | { kind: "json"; items: unknown }
);

export type StageStatus = "start" | "done" | "error";

export interface StageEvent {
  type: "stage";
  stage: string;
  label: string;
  method: string;
  group: string;
  status: StageStatus;
  detail?: Record<string, unknown>;
  payload?: PayloadBlock[];
  elapsed_s?: number;
}

export interface OpenEvent {
  type: "open";
  question: string;
  tender_id?: string | null;
}

export interface FinalEvent {
  type: "final";
  answer: string;
  references: { file_name: string; page_number: string }[];
  confidence: number;
  route?: string | null;
  domain_scope?: string | null;
  graph_domains?: string[] | null;
  // `elapsed_s` is the total pipeline wall-clock time (query in → answer out),
  // set in orchestrator.py's run_pipeline() and forwarded verbatim here.
  debug?: Record<string, unknown> & { elapsed_s?: number };
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type PipelineEvent = OpenEvent | StageEvent | FinalEvent | ErrorEvent;

// ── Stage grouping (mirrors the `group` passed to Stage(...) in orchestrator.py) ──

export const GROUP_ORDER = [
  "understanding",
  "classification",
  "retrieval",
  "graph",
  "fusion",
  "answer",
] as const;

export const GROUP_TITLE: Record<string, string> = {
  understanding: "Query Understanding",
  classification: "Classification & Routing",
  retrieval: "Hybrid Retrieval",
  graph: "Graph Retrieval",
  fusion: "Fusion & Reranking",
  answer: "Answer Generation",
};

const STAGE_ICON: Record<string, string> = {
  query_understanding: "🧠",
  retrieval_need: "🧭",
  domain_prediction: "🗺️",
  graph_scope: "🎯",
  hybrid_rag: "🔎",
  graph_retrieval: "🕸️",
  evidence_fusion: "🧩",
  rerank: "📶",
  answer: "✍️",
};

const GROUP_ICON: Record<string, string> = {
  understanding: "🧠",
  classification: "🧭",
  retrieval: "🔎",
  graph: "🕸️",
  fusion: "🧩",
  answer: "✍️",
};

export function stageIcon(stage: string, group: string): string {
  return STAGE_ICON[stage] || GROUP_ICON[group] || "•";
}

// Turn a stage's `detail` dict (set via Stage.set(**kwargs) server-side) into
// compact "key: value" chips. `reasoning` is rendered as a plain hint chip.
export function detailChips(detail: Record<string, unknown>): { label: string; hint?: boolean }[] {
  const chips: { label: string; hint?: boolean }[] = [];
  for (const [key, value] of Object.entries(detail || {})) {
    if (value === null || value === undefined || value === "") continue;
    if (key === "reasoning") {
      chips.push({ label: String(value), hint: true });
      continue;
    }
    const rendered = Array.isArray(value)
      ? (value.length ? value.join(", ") : "—")
      : String(value);
    chips.push({ label: `${key}: ${rendered}` });
  }
  return chips;
}

// ── SSE client ───────────────────────────────────────────────────────────────

export interface StreamPipelineHandlers {
  onEvent: (ev: PipelineEvent) => void;
  onDone?: () => void;
}

/**
 * POSTs the question to /api/pipeline/stream (proxied by next.config.mjs to the
 * FastAPI backend's `/pipeline/stream`) and forwards each parsed SSE frame to
 * `onEvent`. Resolves once the stream closes (after the `[DONE]` sentinel).
 */
export async function streamPipeline(
  question: string,
  tenderId: string | null,
  { onEvent, onDone }: StreamPipelineHandlers
): Promise<void> {
  const res = await fetch("/api/pipeline/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, tender_id: tenderId }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Pipeline request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIndex: number;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);

        const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        const data = dataLine.slice(5).trim();
        if (data === "[DONE]") continue;

        try {
          onEvent(JSON.parse(data) as PipelineEvent);
        } catch {
          // Ignore malformed frames rather than killing the whole stream.
        }
      }
    }
  } finally {
    onDone?.();
  }
}
