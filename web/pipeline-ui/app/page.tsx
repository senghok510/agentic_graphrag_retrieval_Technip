"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  streamPipeline, StageEvent, PipelineEvent, PayloadBlock,
  GROUP_ORDER, GROUP_TITLE, stageIcon, detailChips,
} from "@/lib/pipeline";

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  pending?: boolean;
  refs?: { file_name: string; page_number: string }[];
  route?: string | null;
  domainScope?: string | null;
  graphDomains?: string[] | null;
  confidence?: number;
  durationS?: number;
}

const SAMPLES = [
  "What is the bid validity period?",                              // textual factoid
  "Who supplies the CO2 compressor?",                             // single-hop
  "Summarize the rotating equipment requirements.",               // aggregation
  "What is the relationship between Package A and Baker Hughes?",  // multi-hop
];

const QUERY_TYPE_LABEL: Record<string, string> = {
  textual_factoid: "Textual Factoid → Hybrid RAG",
  single_hop: "Single-Hop → Local Graph",
  aggregation: "Aggregation → Dual-Level",
  multi_hop: "Multi-hop → Hub-aware PPR",
};

export default function Home() {
  const [tenderId, setTenderId] = useState("ROC_INPEX");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [stages, setStages] = useState<StageEvent[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const traceRef = useRef<HTMLDivElement>(null);

  const toggleStage = useCallback((id: string) => {
    setExpanded((e) => ({ ...e, [id]: !e[id] }));
  }, []);

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);
  useEffect(() => {
    traceRef.current?.scrollTo({ top: traceRef.current.scrollHeight, behavior: "smooth" });
  }, [stages]);

  const upsertStage = useCallback((ev: StageEvent) => {
    setStages((prev) => {
      const i = prev.findIndex((s) => s.stage === ev.stage);
      if (i === -1) return [...prev, ev];
      const next = [...prev];
      next[i] = ev;
      return next;
    });
  }, []);

  const ask = useCallback(async (question: string) => {
    if (!question.trim() || running) return;
    setError(null);
    setStages([]);
    setExpanded({});
    const botId = `bot-${Date.now()}`;
    setMessages((m) => [
      ...m,
      { id: `user-${Date.now()}`, role: "user", text: question },
      { id: botId, role: "bot", text: "", pending: true },
    ]);
    setInput("");
    setRunning(true);

    const onEvent = (ev: PipelineEvent) => {
      if (ev.type === "stage") upsertStage(ev);
      else if (ev.type === "final") {
        setMessages((m) => m.map((msg) => msg.id === botId ? {
          ...msg, pending: false, text: ev.answer, refs: ev.references,
          route: ev.route, domainScope: ev.domain_scope, graphDomains: ev.graph_domains,
          confidence: ev.confidence, durationS: ev.debug?.elapsed_s,
        } : msg));
      } else if (ev.type === "error") {
        setError(ev.message);
        setMessages((m) => m.map((msg) => msg.id === botId ? {
          ...msg, pending: false, text: `⚠️ Pipeline error: ${ev.message}`,
        } : msg));
      }
    };

    try {
      await streamPipeline(question, tenderId || null, { onEvent, onDone: () => setRunning(false) });
    } catch (e: any) {
      setError(String(e?.message || e));
      setRunning(false);
    }
  }, [running, tenderId, upsertStage]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input); }
  };

  const groupsWithStages = GROUP_ORDER.filter((g) => stages.some((s) => s.group === g));

  return (
    <div className="app">
      <header className="top">
        <div className="logo">◆</div>
        <div>
          <h1>Tender Pipeline — Q&amp;A</h1>
          <p>Ask a question and watch the Graph + Vector RAG pipeline run, method by method.</p>
        </div>
        <div className="spacer" />
        <input
          className="tender-select"
          value={tenderId}
          onChange={(e) => setTenderId(e.target.value)}
          placeholder="tender id (ITB)"
          title="ITB / tender id used to scope retrieval"
        />
      </header>

      <div className="grid">
        {/* ── Conversation ── */}
        <section className="panel convo">
          <div className="panel-head">
            Conversation
            <span className="pill">{messages.filter((m) => m.role === "user").length} question(s)</span>
          </div>
          <div className="messages" ref={messagesRef}>
            {messages.length === 0 && (
              <div className="empty">
                <h3>Ask about the tender documents</h3>
                <div>Every question is routed through the pipeline — you&apos;ll see each stage light up on the right.</div>
                <div className="samples">
                  {SAMPLES.map((s) => (
                    <button key={s} className="sample" onClick={() => ask(s)} disabled={running}>{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) => (
              <div key={m.id} className={`msg ${m.role}`}>
                <div className="avatar">{m.role === "user" ? "You" : "◆"}</div>
                <div className="bubble">
                  <div className="who">{m.role === "user" ? "You" : "Assistant"}</div>
                  {m.role === "bot" && !m.pending && m.route && (
                    <div className="route-tags">
                      <span className="tag">{QUERY_TYPE_LABEL[m.route] || m.route}</span>
                      {m.domainScope && (
                        <span className="tag muted">
                          graph: {m.domainScope === "full"
                            ? "full graph"
                            : (m.graphDomains && m.graphDomains.length
                                ? m.graphDomains.join(", ")
                                : m.domainScope)}
                        </span>
                      )}
                      {typeof m.confidence === "number" && (
                        <span className="tag conf">confidence {m.confidence.toFixed(2)}</span>
                      )}
                      {typeof m.durationS === "number" && (
                        <span className="tag muted">{m.durationS.toFixed(1)}s</span>
                      )}
                    </div>
                  )}
                  {m.pending ? (
                    <div className="thinking">
                      <span>Running pipeline</span>
                      <span className="dots"><span>.</span><span>.</span><span>.</span></span>
                    </div>
                  ) : (
                    <div className={`text ${m.role === "bot" ? "answer" : ""}`}>{m.text}</div>
                  )}
                  {m.refs && m.refs.length > 0 && (
                    <div className="refs">
                      <div className="rh">Sources</div>
                      {m.refs.map((r, i) => (
                        <div className="ref" key={i}>
                          <span className="n">[{i + 1}]</span>
                          <span>{r.file_name.replace(/^ROC_INPEX\//, "")} — p.{r.page_number}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask a question about the tender…  (Enter to send, Shift+Enter for newline)"
              rows={1}
            />
            <button className="send" onClick={() => ask(input)} disabled={running || !input.trim()}>
              {running ? "…" : "Send"}
            </button>
          </div>
        </section>

        {/* ── Pipeline inspector ── */}
        <section className="panel inspector">
          <div className="panel-head">
            Pipeline Inspector
            <span className="pill">{running ? "running…" : stages.length ? `${stages.length} stages` : "idle"}</span>
          </div>
          {error && <div className="err-banner">{error}</div>}
          <div className="trace" ref={traceRef}>
            {stages.length === 0 && !running && (
              <div className="trace-empty">
                <div className="big">🧭</div>
                <div>Ask a question to trace the pipeline:<br />understanding → routing → retrieval → fusion → answer.</div>
              </div>
            )}
            {groupsWithStages.map((group) => (
              <div className="group" key={group}>
                <div className="group-label">
                  {GROUP_TITLE[group]}
                  <span className="line" />
                </div>
                {stages.filter((s) => s.group === group).map((s) => {
                  const status = s.status === "start" ? "active" : s.status;
                  const chips = detailChips(s.detail || {});
                  const payload = s.payload || [];
                  const hasPayload = payload.length > 0;
                  const isOpen = !!expanded[s.stage];
                  return (
                    <div className={`stage ${status} ${hasPayload ? "clickable" : ""}`} key={s.stage}>
                      <div className="ico"
                        onClick={hasPayload ? () => toggleStage(s.stage) : undefined}>
                        {status === "active" ? <span className="spinner" />
                          : status === "done" ? "✓"
                          : status === "error" ? "✕"
                          : stageIcon(s.stage, s.group)}
                      </div>
                      <div className="body">
                        <div className="title"
                          onClick={hasPayload ? () => toggleStage(s.stage) : undefined}>
                          <span className="t">{stageIcon(s.stage, s.group)} {s.label}</span>
                          {hasPayload && (
                            <span className="inspect-badge">{isOpen ? "▾" : "▸"} inspect</span>
                          )}
                          {typeof s.elapsed_s === "number" && <span className="elapsed">{s.elapsed_s}s</span>}
                        </div>
                        <span className="method">{s.method}()</span>
                        {chips.length > 0 && (
                          <div className="detail">
                            {chips.map((c, i) => (
                              <span className={`chip ${c.hint ? "hint" : ""}`} key={i}
                                dangerouslySetInnerHTML={{ __html: chipHtml(c.label) }} />
                            ))}
                          </div>
                        )}
                        {hasPayload && isOpen && <StageDetail payload={payload} />}
                      </div>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </section>
      </div>

    </div>
  );
}

// Bold the value half of a "key: value" chip.
function chipHtml(label: string): string {
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const idx = label.indexOf(":");
  if (idx === -1) return esc(label);
  return `${esc(label.slice(0, idx + 1))} <b>${esc(label.slice(idx + 1).trim())}</b>`;
}

// ── Expandable per-stage detail (chunks / relations / entities / text / json) ──
function StageDetail({ payload }: { payload: PayloadBlock[] }) {
  return (
    <div className="stage-detail">
      {payload.map((block, i) => (
        <div className="detail-block" key={i}>
          <div className="detail-block-title">
            {block.title}
            {Array.isArray(block.items) && <span className="count">{block.items.length}</span>}
          </div>
          {block.kind === "chunks" && <ChunkList items={block.items} />}
          {block.kind === "relations" && <RelationList items={block.items} />}
          {block.kind === "entities" && <EntityList items={block.items} />}
          {block.kind === "text" && <pre className="detail-text">{String(block.items)}</pre>}
          {block.kind === "json" && (
            <pre className="detail-text">{JSON.stringify(block.items, null, 2)}</pre>
          )}
        </div>
      ))}
    </div>
  );
}

function ChunkList({ items }: { items: any[] }) {
  if (!items?.length) return <div className="detail-empty">— none —</div>;
  return (
    <div className="chunk-list">
      {items.map((c, i) => (
        <div className="chunk-card" key={i}>
          <div className="chunk-head">
            <span className="chunk-file">📄 {c.file_name}</span>
            <span className="chunk-meta">p.{c.page_number}</span>
            {c.source && <span className="chunk-src">{c.source}</span>}
            {typeof c.score === "number" && <span className="chunk-score">{c.score}</span>}
          </div>
          {c.text && <div className="chunk-text">{c.text}</div>}
        </div>
      ))}
    </div>
  );
}

function RelationList({ items }: { items: any[] }) {
  if (!items?.length) return <div className="detail-empty">— none —</div>;
  return (
    <div className="rel-list">
      {items.map((r, i) => (
        <div className="rel-row" key={i}>
          <div className="rel-triple">
            <span className="rel-ent">{r.subject}</span>
            <span className="rel-pred">—{r.label}→</span>
            <span className="rel-ent">{r.object}</span>
            {typeof r.score === "number" && <span className="rel-score">{r.score}</span>}
            {r.channel && <span className="rel-chan">{r.channel}</span>}
          </div>
          {r.description && <div className="rel-desc">{r.description}</div>}
        </div>
      ))}
    </div>
  );
}

function EntityList({ items }: { items: any[] }) {
  if (!items?.length) return <div className="detail-empty">— none —</div>;
  return (
    <div className="ent-list">
      {items.map((e, i) => (
        <span className="ent-pill" key={i}>
          {e.name}
          {e.category && <span className="ent-cat">{e.category}</span>}
          {typeof e.score === "number" && <span className="ent-score">{e.score}</span>}
        </span>
      ))}
    </div>
  );
}
