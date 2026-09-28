"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ConversationPanel,
  Message,
} from "@/components/conversation-panel";
import { PipelineInspector } from "@/components/pipeline-inspector";
import { PipelineEvent, StageEvent, streamPipeline } from "@/lib/pipeline";

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

  useEffect(() => {
    messagesRef.current?.scrollTo({
      top: messagesRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  useEffect(() => {
    traceRef.current?.scrollTo({
      top: traceRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [stages]);

  const upsertStage = useCallback((event: StageEvent) => {
    setStages((current) => {
      const index = current.findIndex((stage) => stage.stage === event.stage);
      if (index === -1) return [...current, event];
      const updated = [...current];
      updated[index] = event;
      return updated;
    });
  }, []);

  const applyPipelineEvent = useCallback((event: PipelineEvent, botId: string) => {
    if (event.type === "stage") {
      upsertStage(event);
      return;
    }
    if (event.type === "final") {
      setMessages((current) => current.map((message) => message.id === botId ? {
        ...message,
        pending: false,
        text: event.answer,
        refs: event.references,
        route: event.route,
        domainScope: event.domain_scope,
        graphDomains: event.graph_domains,
        confidence: event.confidence,
        durationS: event.debug?.elapsed_s,
      } : message));
      return;
    }
    if (event.type === "error") {
      setError(event.message);
      setMessages((current) => current.map((message) => message.id === botId ? {
        ...message,
        pending: false,
        text: `⚠️ Pipeline error: ${event.message}`,
      } : message));
    }
  }, [upsertStage]);

  const ask = useCallback(async (rawQuestion: string) => {
    const question = rawQuestion.trim();
    if (!question || running) return;

    const requestId = Date.now();
    const botId = `bot-${requestId}`;
    setError(null);
    setStages([]);
    setExpanded({});
    setMessages((current) => [
      ...current,
      { id: `user-${requestId}`, role: "user", text: question },
      { id: botId, role: "bot", text: "", pending: true },
    ]);
    setInput("");
    setRunning(true);

    try {
      await streamPipeline(question, tenderId || null, {
        onEvent: (event) => applyPipelineEvent(event, botId),
        onDone: () => setRunning(false),
      });
    } catch (caught: unknown) {
      const message = caught instanceof Error ? caught.message : String(caught);
      setError(message);
      setMessages((current) => current.map((item) => item.id === botId ? {
        ...item,
        pending: false,
        text: `⚠️ Pipeline error: ${message}`,
      } : item));
      setRunning(false);
    }
  }, [applyPipelineEvent, running, tenderId]);

  const toggleStage = useCallback((stage: string) => {
    setExpanded((current) => ({ ...current, [stage]: !current[stage] }));
  }, []);

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
          onChange={(event) => setTenderId(event.target.value)}
          placeholder="tender id (ITB)"
          title="ITB / tender id used to scope retrieval"
        />
      </header>

      <div className="grid">
        <ConversationPanel
          input={input}
          messages={messages}
          messagesRef={messagesRef}
          running={running}
          onInputChange={setInput}
          onAsk={ask}
        />
        <PipelineInspector
          error={error}
          expanded={expanded}
          running={running}
          stages={stages}
          traceRef={traceRef}
          onToggleStage={toggleStage}
        />
      </div>
    </div>
  );
}
