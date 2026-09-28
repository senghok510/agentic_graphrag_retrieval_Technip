import type { KeyboardEvent, RefObject } from "react";

export interface Message {
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

interface ConversationPanelProps {
  input: string;
  messages: Message[];
  messagesRef: RefObject<HTMLDivElement>;
  running: boolean;
  onInputChange: (value: string) => void;
  onAsk: (question: string) => void;
}

const SAMPLES = [
  "What is the bid validity period?",
  "Who supplies the CO2 compressor?",
  "Summarize the rotating equipment requirements.",
  "What is the relationship between Package A and Baker Hughes?",
];

const QUERY_TYPE_LABEL: Record<string, string> = {
  textual_factoid: "Textual Factoid → Hybrid RAG",
  single_hop: "Single-Hop → Local Graph",
  aggregation: "Aggregation → Dual-Level",
  multi_hop: "Multi-hop → Hub-aware PPR",
};

export function ConversationPanel({
  input,
  messages,
  messagesRef,
  running,
  onInputChange,
  onAsk,
}: ConversationPanelProps) {
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onAsk(input);
    }
  };

  return (
    <section className="panel convo">
      <div className="panel-head">
        Conversation
        <span className="pill">
          {messages.filter((message) => message.role === "user").length} question(s)
        </span>
      </div>

      <div className="messages" ref={messagesRef}>
        {messages.length === 0 && (
          <div className="empty">
            <h3>Ask about the tender documents</h3>
            <div>
              Every question is routed through the pipeline — each stage appears on the right.
            </div>
            <div className="samples">
              {SAMPLES.map((sample) => (
                <button
                  key={sample}
                  className="sample"
                  onClick={() => onAsk(sample)}
                  disabled={running}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>

      <div className="composer">
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about the tender…  (Enter to send, Shift+Enter for newline)"
          rows={1}
        />
        <button
          className="send"
          onClick={() => onAsk(input)}
          disabled={running || !input.trim()}
        >
          {running ? "…" : "Send"}
        </button>
      </div>
    </section>
  );
}

function MessageBubble({ message }: { message: Message }) {
  return (
    <div className={`msg ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : "◆"}</div>
      <div className="bubble">
        <div className="who">{message.role === "user" ? "You" : "Assistant"}</div>
        {message.role === "bot" && !message.pending && message.route && (
          <div className="route-tags">
            <span className="tag">{QUERY_TYPE_LABEL[message.route] || message.route}</span>
            {message.domainScope && (
              <span className="tag muted">
                graph: {formatGraphScope(message.domainScope, message.graphDomains)}
              </span>
            )}
            {typeof message.confidence === "number" && (
              <span className="tag conf">confidence {message.confidence.toFixed(2)}</span>
            )}
            {typeof message.durationS === "number" && (
              <span className="tag muted">{message.durationS.toFixed(1)}s</span>
            )}
          </div>
        )}

        {message.pending ? (
          <div className="thinking">
            <span>Running pipeline</span>
            <span className="dots"><span>.</span><span>.</span><span>.</span></span>
          </div>
        ) : (
          <div className={`text ${message.role === "bot" ? "answer" : ""}`}>
            {message.text}
          </div>
        )}

        {message.refs && message.refs.length > 0 && (
          <div className="refs">
            <div className="rh">Sources</div>
            {message.refs.map((reference, index) => (
              <div className="ref" key={`${reference.file_name}-${reference.page_number}-${index}`}>
                <span className="n">[{index + 1}]</span>
                <span>
                  {reference.file_name.replace(/^ROC_INPEX\//, "")} — p.{reference.page_number}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function formatGraphScope(scope: string, domains?: string[] | null): string {
  if (scope === "full") return "full graph";
  return domains?.length ? domains.join(", ") : scope;
}
