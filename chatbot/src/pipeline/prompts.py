"""Prompt templates for the retrieval pipeline.

Copied verbatim from ``ppr.py`` / ``evaluate_lightrag.py`` so behaviour is
identical; centralised here so every stage module shares one source of truth.
"""

# ── Query understanding ──────────────────────────────────────────────────────

HYDE_PROMPT = """Write a factual, dense, and highly professional paragraph in tendering domain that would perfectly answer the following query.
This paragraph will be used for vector similarity search, so include technical terminology and specific details that would likely appear in the source documents.

Query: {query}

Factual Paragraph:"""

QUERY_EXPANSION_PROMPT_COMPLEX = """
You are an expert in ITB (Invitation to Tender/Bid) and EPC (Engineering, Procurement, and Construction) procurement.

Your task is to rewrite a user query into search terms that yield precise retrieval from technical tender documents.

### GUIDELINES:
1. "rephrased_questions": Rewrite the query 3 times using professional procurement syntax.
2. "keywords": Provide 6-10 specific industry terms.
   - DO NOT use generic terms like 'things', 'stuff', or 'data'.
   - USE specific EPC/Contractual roles
   - USE specific document register terms

### EXAMPLE:
Input: "What are the requirements for the final project report?"
Output: {{
  "rephrased_questions": [
    "Specify the contractual obligations for the submission of the final project completion report.",
    "What are the technical specifications and documentation standards required for the final project deliverable?",
    "Identify the milestones and due dates associated with the final reporting requirements in the scope of work."
  ],
  "keywords": ["final report", "deliverable", "technical specification", "scope of work", "milestone", "completion certificate", "documentation standard", "compliance"]
}}

### YOUR TASK:
Query: {query}

Produce a structured JSON response. Do not include any explanatory text outside the JSON.
"""

QUERY_EXPANSION_PROMPT_SIMPLE = """

Rewrite and rephrase the following query into alternative forms that yield better retrieval results from tender documents.
Each rephrased version must preserve the original intent but use different vocabulary — more formal language, technical terminology, abbreviations, or synonyms commonly found in procurement documentation.

Query: {query}

Produce a structured JSON response with exactly one field:

1. "rephrased_questions":
   Example — Original: "My RAG is too slow, what do I do?"
   Good rewrites: ["Techniques to reduce latency in RAG systems.", "Optimizing performance of Retrieval-Augmented Generation pipelines.", "How to improve RAG system response time?"]

2. "keywords": A list of 3-5 strings — individual keywords (not phrases) that cover the full range of meanings this query could have in an ITB context. Include synonyms, abbreviations, and related procurement/engineering terms.

Return only valid JSON matching this schema. Do not include any explanatory text outside the JSON."""


GRAPH_COMPLEXITY_PROMPT = """
Query to analyze: "{question}"

You have TWO tasks. Do BOTH and return ONE JSON object.

============================================================
TASK A — STEP-BY-STEP COMPLEXITY ANALYSIS
============================================================

 STEP-BY-STEP ANALYSIS PROCESS:

    Step 1: Count the number of distinct questions
    - Look for conjunctions: "and", "or", "also", "?"
    - Look for multiple question words: "what...what", "who...what"
    - Result: Does this ask ONE question or MULTIPLE questions?

    Step 2: Check for aggregation/list/summarization indicators
    - Words like: "all", "list", "requirements" (plural), "criteria" (plural),
      "summarize", "summarise", "overview", "describe", "short paragraph",
      "main", "key", "objectives", "goals", "scope", "purpose"
    - If present, likely COMPLEX because the answer requires synthesis or aggregation.

    Step 3: Check for comparison/relationship indicators
    - Words like: "compare", "difference", "versus", "relationship", "similarity",
      "connect", "link", "tie into", "depend", "interact"
    - If present, definitely COMPLEX.

    Step 4: Check for process/sequential indicators
    - Words like: "process", "procedure", "steps", "how to", "workflow",
      "before", "after", "during", "sequence"
    - If present, definitely COMPLEX.

    Step 5: Final decision
    - If 2+ questions OR aggregation/list/summarization OR comparison/relationship OR process then COMPLEX.
    - If 1 specific atomic fact, value, date, name, clause, party, or rule then SIMPLE.
    - A request for a short answer is NOT automatically SIMPLE.
      If the answer is a synthesized paragraph or overview, classify it as COMPLEX.

============================================================
TASK B — GRAPH ROUTING HINTS
============================================================

Extract lists from the question. These drive graph traversal.

1. entity_hints — specific named things appearing in the question text: core noun,
   proper nouns, exhibit refs ("Exhibit L"), clause numbers ("Clause 12.4"),
   document names ("HSES management plan"), system tags ("CCS Unit 1"),
   organisation names ("INPEX"), etc. For a query, there should be one or more
   entity_hints that capture the core of the question.

2. category_hints — canonical entity categories the question is asking about.
   MUST be exact strings from this closed list (drop anything not on the list):
   {entity_categories}

3. relation_hints — free-form relationship intent hints implied by the question.
   Use short snake_case phrases that describe the kind of relationship or traversal
   the question is asking for.

4. high_level_keywords — overarching concepts or themes — the TYPE of relationship or
   theme the question is about (snake_case, lowercase, NO entity names or numbers),
   e.g. schedule_dependency, payment_terms, compliance_obligation, scope_inclusion.

============================================================
TASK C — Query decomposition
============================================================

Decompose the query into sub-questions if the question involves finding information from different sources or documents. The sub-questions are the factoid questions that retrieve local different context to support the original question.

============================================================
EXAMPLES
============================================================

Query: "What is the project currency?"

Output:
{{
  "is_complex": false,
  "reasoning": "The query asks for one specific atomic value, the project currency, so it is simple fact retrieval.",
  "sub_questions": ["What is the project currency?"],
  "entity_hints": ["project currency"],
  "category_hints": ["amount"],
  "relation_hints": [],
  "high_level_keywords": []
}}

Query: "List all the technical requirements that apply to CCS Unit 1"

Output:
{{
  "is_complex": true,
  "reasoning": "The query asks to list multiple requirements applying to a named system, so it requires aggregation across relevant requirement evidence.",
  "sub_questions": ["What technical requirements apply to CCS Unit 1?"],
  "entity_hints": ["technical requirements", "CCS Unit 1"],
  "category_hints": ["requirement", "system"],
  "relation_hints": ["applies_to"],
  "high_level_keywords": ["technical_requirements", "compliance"]
}}

Query: "How does the Document Deliverables Register in Exhibit A13 connect to the Project Controls requirements in Exhibit L?"

Output:
{{
  "is_complex": true,
  "reasoning": "The query asks for the relationship between named documents and requirements across exhibits, so it needs relational/multi-hop reasoning.",
  "sub_questions": [
    "What is the Document Deliverables Register in Exhibit A13?",
    "What are the Project Controls requirements in Exhibit L?"
  ],
  "entity_hints": ["Document Deliverables Register", "Exhibit A13", "Project Controls requirements", "Exhibit L"],
  "category_hints": ["deliverable", "document", "requirement"],
  "relation_hints": ["connects_to", "references", "applies_to"],
  "high_level_keywords": ["document_deliverables", "project_controls", "document_reference"]
}}

============================================================
OUTPUT
============================================================

Return STRICT JSON matching exactly:
{{
  "is_complex": true,
  "reasoning": "...",
  "sub_questions": ["..."],
  "entity_hints": ["..."],
  "category_hints": ["..."],
  "relation_hints": ["..."],
  "high_level_keywords": ["..."]
}}
""".strip()


# ── Query-type classification (factoid / relational / summarization) ─────────

STRATEGY_SYSTEM_PROMPT = (
    "You classify tender/ITB graph questions by the kind of retrieval they need. "
    "Return strict JSON only."
)

STRATEGY_USER_PROMPT = """Classify the question into ONE strategy.

- factoid       : a single atomic fact / value / date / clause / party (Simple Query).
- relational    : the connection between specific named entities, or multi-hop reasoning (Relational / Multi-hop Query).
- summarization : aggregation / list / overview across many entities or a theme (Aggregation / List Query).

Question: {question}

Return STRICT JSON: {{"strategy": "factoid|relational|summarization", "reasoning": "one line"}}
"""


# ── v2: Retrieval Need Classification (HOW to retrieve) ──────────────────────

RETRIEVAL_NEED_SYSTEM_PROMPT = (
    "You classify tender/ITB questions by the KIND of retrieval they need. "
    "This is independent of any domain. Return strict JSON only."
)

RETRIEVAL_NEED_USER_PROMPT = """Classify the question into EXACTLY ONE retrieval need.

- textual_factoid : the answer is an explicit value/fact stated verbatim in the text
                    (a number, date, percentage, amount, name, clause). No graph needed.
                    e.g. "What is the bid validity period?", "What is the performance bond amount?"
- single_hop      : one graph edge answers it — pattern (s, p, *) or (*, p, o):
                    one known entity + one predicate, asking for the other endpoint.
                    e.g. "Who supplies the CO2 compressor?", "Which equipment is supplied by Baker Hughes?"
- aggregation     : collect / summarize / list many related facts about a topic or domain.
                    e.g. "Summarize rotating equipment requirements.", "List all procurement obligations."
- multi_hop       : discover the connection BETWEEN two entities, or traverse several edges —
                    pattern (s, *, o) or anything needing a path.
                    e.g. "What is the relationship between Package A and Baker Hughes?"

Question: {question}

Return STRICT JSON:
{{"need": "textual_factoid|single_hop|aggregation|multi_hop", "query_form": "(s,p,*)|(*,p,o)|(s,*,o)|none", "reasoning": "one line"}}
"""


# ── v2: Graph Domain Prediction (WHERE the graph searches) ───────────────────

DOMAIN_PREDICTION_SYSTEM_PROMPT = (
    "You predict which engineering domain(s) a GRAPH search should be scoped to for an "
    "EPC/ITB tender question. This scopes ONLY graph retrieval — full-text search always "
    "covers everything. Return strict JSON only."
)

DOMAIN_PREDICTION_USER_PROMPT = """Decide how to scope the GRAPH search for this question.

- scope = "single" : the question clearly concerns ONE engineering domain.
- scope = "multi"  : it spans a few related domains (list them).
- scope = "full"   : it is general, cross-cutting, or you are unsure — search the entire graph.

Only use domain names from this list (drop anything not on it). When scope="full", return an empty domains list.

Candidate domains:
{domain_block}

Question: {question}

Return STRICT JSON:
{{"scope": "single|multi|full", "domains": ["<exact domain name>", ...], "reasoning": "one line"}}
"""


# ── High-level keyword extraction (LightRAG global channel) ──────────────────

QUERY_HL_SYS = """You are an expert in tendering and EPC/ITB contract documents.
Extract the high-level themes a question is about — overarching concepts and themes,
NOT specific entities, clause numbers, or document names."""

QUERY_HL_USER = """Extract 2-5 high-level theme keywords describing what KIND of
relationship the question is asking about, in the same style as:
schedule_dependency, payment_terms, compliance_obligation, liability_allocation,
scope_inclusion, test_sequencing, approval_workflow, document_reference.
Return snake_case, lowercase. No entity names or numbers.

Question: {question}

Return STRICT JSON: {{"high_level_keywords": ["theme_one", "theme_two"]}}
"""


# ── Answer generation ────────────────────────────────────────────────────────

SYNTHESIZE_ANSWER_SYSTEM_PROMPT = """
You are a senior specialist in tendering, ITB analysis, and EPC oil & gas contract documents.
Answer the user's question using ONLY the provided context. Do not use outside knowledge.
If the context does not contain enough information to answer, say so explicitly instead of guessing.
"""

SYNTHESIZE_ANSWER_USER_PROMPT = """
Question:
{question}

Context:
{context}

Write a clear, concise answer to the question, grounded strictly in the context above.
""".strip()


def detailed_answer_prompt(question: str, strategy: str, complexity_guidance: str,
                           total_sources: int, numbered_knowledge_base: str) -> str:
    """The grounded, cited answer prompt used by ``answer.detailed_respond``."""
    return f"""You are an expert in tendering and procurement documents. Answer the user's question accurately and completely using ONLY the provided context.

User Question: {question}
Routing Guidance: strategy={strategy}
{complexity_guidance}

### AVAILABLE SOURCES ({total_sources} total):
The context below contains EXACTLY {total_sources} sources, labeled [Source 1] through [Source {total_sources}].
You MUST NOT reference any source number beyond [Source {total_sources}].
You do NOT need to cite all {total_sources} sources — only cite the ones that are relevant to your answer.
If a claim cannot be attributed to a labeled source, omit the claim or state "Information not found in available documentation."

### MANDATORY RULES:
1. Inline Citations: Place a citation [Source X] immediately after each claim or sentence that uses information from the context.
2. Strictness: Use ONLY information from the context.
3. Structure: Use clear headings, bullet points, and a professional tone.
4. Entity Analysis: Include a Key Entities/Requirements section if relevant.
5. Aesthetics: Use relevant emojis (📝, 📊, ✅, 📌, 💡).
6. Discrepancies: If there are discrepancies between documents, point them out.
7. Start each major heading with one relevant emoji.
8. Use only these emojis: 📝, 📊, ✅, 📌, 💡.
Return STRICT JSON with fields:
- answer: string
- references: []
- confidence_score: number

Context:
{numbered_knowledge_base}
""".strip()
