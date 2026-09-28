import type { RefObject } from "react";
import {
  detailChips,
  EntityItem,
  GROUP_ORDER,
  GROUP_TITLE,
  PayloadBlock,
  RelationItem,
  stageIcon,
  StageEvent,
  ChunkItem,
} from "@/lib/pipeline";

interface PipelineInspectorProps {
  error: string | null;
  expanded: Record<string, boolean>;
  running: boolean;
  stages: StageEvent[];
  traceRef: RefObject<HTMLDivElement>;
  onToggleStage: (stage: string) => void;
}

export function PipelineInspector({
  error,
  expanded,
  running,
  stages,
  traceRef,
  onToggleStage,
}: PipelineInspectorProps) {
  const groups = GROUP_ORDER.filter((group) => stages.some((stage) => stage.group === group));

  return (
    <section className="panel inspector">
      <div className="panel-head">
        Pipeline Inspector
        <span className="pill">
          {running ? "running…" : stages.length ? `${stages.length} stages` : "idle"}
        </span>
      </div>
      {error && <div className="err-banner">{error}</div>}
      <div className="trace" ref={traceRef}>
        {stages.length === 0 && !running && (
          <div className="trace-empty">
            <div className="big">🧭</div>
            <div>
              Ask a question to trace the pipeline:<br />
              understanding → routing → retrieval → fusion → answer.
            </div>
          </div>
        )}

        {groups.map((group) => (
          <div className="group" key={group}>
            <div className="group-label">
              {GROUP_TITLE[group]}
              <span className="line" />
            </div>
            {stages
              .filter((stage) => stage.group === group)
              .map((stage) => (
                <StageCard
                  key={stage.stage}
                  stage={stage}
                  expanded={Boolean(expanded[stage.stage])}
                  onToggle={() => onToggleStage(stage.stage)}
                />
              ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function StageCard({
  stage,
  expanded,
  onToggle,
}: {
  stage: StageEvent;
  expanded: boolean;
  onToggle: () => void;
}) {
  const status = stage.status === "start" ? "active" : stage.status;
  const chips = detailChips(stage.detail || {});
  const payload = stage.payload || [];
  const hasPayload = payload.length > 0;
  const toggleProps = hasPayload ? { onClick: onToggle } : {};

  return (
    <div className={`stage ${status} ${hasPayload ? "clickable" : ""}`}>
      <div className="ico" {...toggleProps}>
        {status === "active" ? <span className="spinner" />
          : status === "done" ? "✓"
          : status === "error" ? "✕"
          : stageIcon(stage.stage, stage.group)}
      </div>
      <div className="body">
        <div className="title" {...toggleProps}>
          <span className="t">{stageIcon(stage.stage, stage.group)} {stage.label}</span>
          {hasPayload && (
            <span className="inspect-badge">{expanded ? "▾" : "▸"} inspect</span>
          )}
          {typeof stage.elapsed_s === "number" && (
            <span className="elapsed">{stage.elapsed_s}s</span>
          )}
        </div>
        <span className="method">{stage.method}()</span>
        {chips.length > 0 && (
          <div className="detail">
            {chips.map((chip, index) => <DetailChip key={index} {...chip} />)}
          </div>
        )}
        {hasPayload && expanded && <StageDetail payload={payload} />}
      </div>
    </div>
  );
}

function DetailChip({ label, hint }: { label: string; hint?: boolean }) {
  const separator = label.indexOf(":");
  if (separator === -1) return <span className={`chip ${hint ? "hint" : ""}`}>{label}</span>;
  return (
    <span className={`chip ${hint ? "hint" : ""}`}>
      {label.slice(0, separator + 1)} <b>{label.slice(separator + 1).trim()}</b>
    </span>
  );
}

function StageDetail({ payload }: { payload: PayloadBlock[] }) {
  return (
    <div className="stage-detail">
      {payload.map((block, index) => (
        <div className="detail-block" key={`${block.kind}-${block.title}-${index}`}>
          <div className="detail-block-title">
            {block.title}
            {Array.isArray(block.items) && <span className="count">{block.items.length}</span>}
          </div>
          {block.kind === "chunks" && <ChunkList items={block.items} />}
          {block.kind === "relations" && <RelationList items={block.items} />}
          {block.kind === "entities" && <EntityList items={block.items} />}
          {block.kind === "text" && <pre className="detail-text">{block.items}</pre>}
          {block.kind === "json" && (
            <pre className="detail-text">{JSON.stringify(block.items, null, 2)}</pre>
          )}
        </div>
      ))}
    </div>
  );
}

function ChunkList({ items }: { items: ChunkItem[] }) {
  if (!items.length) return <div className="detail-empty">— none —</div>;
  return (
    <div className="chunk-list">
      {items.map((chunk, index) => (
        <div className="chunk-card" key={`${chunk.file_name}-${chunk.page_number}-${index}`}>
          <div className="chunk-head">
            <span className="chunk-file">📄 {chunk.file_name}</span>
            <span className="chunk-meta">p.{chunk.page_number}</span>
            {chunk.source && <span className="chunk-src">{chunk.source}</span>}
            {typeof chunk.score === "number" && <span className="chunk-score">{chunk.score}</span>}
          </div>
          {chunk.text && <div className="chunk-text">{chunk.text}</div>}
        </div>
      ))}
    </div>
  );
}

function RelationList({ items }: { items: RelationItem[] }) {
  if (!items.length) return <div className="detail-empty">— none —</div>;
  return (
    <div className="rel-list">
      {items.map((relation, index) => (
        <div className="rel-row" key={`${relation.subject}-${relation.label}-${relation.object}-${index}`}>
          <div className="rel-triple">
            <span className="rel-ent">{relation.subject}</span>
            <span className="rel-pred">—{relation.label}→</span>
            <span className="rel-ent">{relation.object}</span>
            {typeof relation.score === "number" && <span className="rel-score">{relation.score}</span>}
            {relation.channel && <span className="rel-chan">{relation.channel}</span>}
          </div>
          {relation.description && <div className="rel-desc">{relation.description}</div>}
        </div>
      ))}
    </div>
  );
}

function EntityList({ items }: { items: EntityItem[] }) {
  if (!items.length) return <div className="detail-empty">— none —</div>;
  return (
    <div className="ent-list">
      {items.map((entity, index) => (
        <span className="ent-pill" key={`${entity.name}-${index}`}>
          {entity.name}
          {entity.category && <span className="ent-cat">{entity.category}</span>}
          {typeof entity.score === "number" && <span className="ent-score">{entity.score}</span>}
        </span>
      ))}
    </div>
  );
}
