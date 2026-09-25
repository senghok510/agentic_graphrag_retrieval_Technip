import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from langchain_core.tools import tool
from ..config.appSettings import get_settings,embed_document_large_model,embed_document_neo4j


logger = logging.getLogger("agent_flow.neo4j_service")

_settings = None
_driver = None

def _get_settings():
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings

def get_driver():
    global _driver
    if _driver is None:
        settings = _get_settings()
        NEO4J_URI = getattr(settings, "NEO4J_URI")
        NEO4J_USER = getattr(settings, "NEO4J_USER")
        NEO4J_PASSWORD = getattr(settings, "NEO4J_PASSWORD")
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

def neo4j_run_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Exécute une requête Cypher via le protocole Bolt.
    """
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            records = [record.data() for record in result]
            if params and 'ITB_ID' in params:
                logger.debug(f"Neo4j query: ITB_ID={params['ITB_ID']}, {len(records)} records")
            return records
    except Exception as e:
        logger.error(f"Neo4j query error: {e}")
        return []

def neo4j_vector_search(
    question_embedding: List[float],
    tender_id: str,
    top_k: int = 10,
    index_name: str = "document_chunks"
) -> List[Dict[str, Any]]:
    """
    Recherche vectorielle optimisée avec enrichissement de contexte.
    """
    vector_search_query = """
    CALL db.index.vector.queryNodes($index_name, $top_k_search, $question_embedding) 
    YIELD node, score
    
    MATCH (node)-[:PART_OF]->(d:Document)
    MATCH (d)-[:BELONGS_TO]->(i:ITB {ITB_ID: $ITB_ID})

    OPTIONAL MATCH (d)-[:HAS_LOW_LEVEL]->(domain:LowLevelDomain)
    OPTIONAL MATCH (d)-[:HAS_HIGH_DOMAIN]->(hdomain:HighLevelDomain)
    
    // Context retrieval (Previous / Next chunks)
    OPTIONAL MATCH (prev:Chunk)-[:PART_OF]->(d) WHERE prev.sequenceNo = node.sequenceNo - 1
    OPTIONAL MATCH (next:Chunk)-[:PART_OF]->(d) WHERE next.sequenceNo = node.sequenceNo + 1

    RETURN 
    i.ITB_ID AS itb_name,
    node.sequenceNo AS Chunk_SeqNo,
    node.text AS current_chunk_text,
    node.ChunkID AS CurrentChunkID,
    prev.text AS prev_chunk_text,
    next.text AS next_chunk_text,
    d.docURL AS document_filename,
    domain.domainName AS low_level_domain,
    score
    ORDER BY score DESC
    LIMIT $top_k;
    """

    params = {
        "question_embedding": question_embedding,
        "index_name": index_name,
        "top_k": top_k,
        "top_k_search": 100,
        "ITB_ID": tender_id
    }

    results = neo4j_run_query(vector_search_query, params)
    
    if results:
        logger.debug(f"Vector search: {len(results)} chunks, ITB_ID={tender_id}")
    else:
        logger.warning(f"Vector search: 0 results, ITB_ID={tender_id}")
    return results


@tool
def neo4j_graph_tool(query: str, tender_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    LangChain tool: vector search on Neo4j graph database.
    """
    try:

        query_embedding = embed_document_neo4j(query)
        records = neo4j_vector_search(query_embedding, tender_id, top_k=top_k)
        
        formatted = [{
            "page_chunk": r.get('current_chunk_text', ''),
            "file_name": r.get('document_filename', 'N/A'),
            "page_number": str(r.get('Chunk_SeqNo', 'N/A')),
            "score": float(r.get('score', 0.0)),
            "CurrentChunkID": r.get('CurrentChunkID', ''),
            "context_chunks": {
                "previous": r.get('prev_chunk_text', ''),
                "next": r.get('next_chunk_text', '')
            },
            "source": "graph_db"
        } for r in records]
        
        logger.debug(f"Neo4j graph tool: {len(formatted)} results")
        return formatted

    except Exception as e:
        logger.error(f"Neo4j graph tool failed: {e}")
        return []

# @tool
# def neo4j_fulltext_search(keyword: str, tender_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
#     """
#     Search for specific keywords or codes in text or SequenceNo properties using substring scan (CONTAINS).
#     Useful for exact numeric codes like '3.1.2' or 'TB-0010'.
#     """
#     fulltext_query = """
#     CALL db.index.fulltext.queryNodes('chunk_fulltext', $keyword)
#     YIELD node AS c, score
#     MATCH (c)-[:PART_OF]->(d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
#     RETURN c.text AS chunk_text, d.docURL AS filename, c.sequenceNo AS page_number, c.ChunkID AS chunk_id, score
#     ORDER BY score DESC
#     LIMIT $top_k
#     """
#     results = neo4j_run_query(fulltext_query, {"keyword": keyword, "tender_id": tender_id, "top_k": top_k})

#     # Fallback to CONTAINS for numeric/code searches that Lucene tokenises poorly
#     if not results:
#         logger.debug(f"Lucene index returned 0 results for '{keyword}', falling back to CONTAINS scan")
        
#         fallback_query = """
#         MATCH (c:Chunk)-[:PART_OF]->(d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
#         WHERE c.text CONTAINS $keyword OR toString(c.sequenceNo) CONTAINS $keyword
#         RETURN c.text AS chunk_text, d.docURL AS filename, c.sequenceNo AS page_number, c.ChunkID AS chunk_id, 0.0 AS score
#         LIMIT $top_k
#         """
#         results = neo4j_run_query(fallback_query, {"keyword": keyword, "tender_id": tender_id, "top_k": top_k})

#     return results

@tool
def neo4j_expand_context_by_ids(chunk_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieve immediate neighbors and related document metadata for Chunk IDs.
    """
    if not chunk_ids:
        return []
        
    query = """
    MATCH (node:Chunk)
    WHERE node.ChunkID IN $chunk_ids
    MATCH (node)-[:PART_OF]->(d:Document)
    OPTIONAL MATCH (prev:Chunk)-[:PART_OF]->(d) WHERE prev.sequenceNo = node.sequenceNo - 1
    OPTIONAL MATCH (next:Chunk)-[:PART_OF]->(d) WHERE next.sequenceNo = node.sequenceNo + 1
    OPTIONAL MATCH (d)-[:HAS_LOW_LEVEL]->(domain:LowLevelDomain)
    
    RETURN 
    node.ChunkID as chunk_id,
    node.text as current_text,
    node.sequenceNo as seq_no,
    prev.text as prev_text,
    next.text as next_text,
    d.docURL as filename,
    domain.domainName as domain
    """
    return neo4j_run_query(query, {"chunk_ids": chunk_ids})

@tool
def neo4j_get_document_relationships(filename: str, tender_id: str) -> List[Dict[str, Any]]:
    """
    Find documents related to a specific document.
    """
    query = """
    MATCH (d:Document {docURL: $filename})-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[r]-(related:Document)
    RETURN type(r) as relationship, related.docURL as related_file
    LIMIT 20
    """
    return neo4j_run_query(query, {"filename": filename, "tender_id": tender_id})

@tool
def neo4j_get_itb_hierarchy(tender_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve the structure for a given ITB.
    """
    query = """
    MATCH (i:ITB {ITB_ID: $tender_id})<-[:BELONGS_TO]-(d:Document)
    OPTIONAL MATCH (d)-[:HAS_LOW_LEVEL]->(ld:LowLevelDomain)
    OPTIONAL MATCH (d)-[:HAS_HIGH_DOMAIN]->(hd:HighLevelDomain)
    RETURN 
    hd.domainName as high_domain,
    ld.domainName as low_domain,
    collect(d.docURL) as documents
    ORDER BY high_domain, low_domain
    """
    return neo4j_run_query(query, {"tender_id": tender_id})

def get_all_documents_for_itb(tender_id: str) -> List[str]:
    """
    Retrieve a list of all document filenames for a given ITB.
    """
    query = """
    MATCH (i:ITB {ITB_ID: $tender_id})<-[:BELONGS_TO]-(d:Document)
    RETURN d.docURL as filename
    """
    results = neo4j_run_query(query, {"tender_id": tender_id})
    return [r['filename'] for r in results if r.get('filename')]

def close_neo4j_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        logger.info("Neo4j Bolt connection closed.")


### ask specifcally for a pdf file:
## work like in chatgpt



# @tool
# def neo4j_graph_tool_filtered(
#     query: str,
#     itb_id: str,
#     top_k: int = 10,
#     high_domain: Optional[str] = None,
#     low_domain: Optional[str] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     LangChain tool: vector search with Cypher 25 in-index domain filtering.

#     Use this instead of neo4j_graph_tool when the user's question is scoped
#     to a specific discipline or sub-discipline (detected by query_analyzer_agent).

#     Args:
#         query:       Natural language question.
#         itb_id:      ITB tenant scope (always required).
#         top_k:       Number of chunks to return.
#         high_domain: Discipline filter e.g. 'TECHNICAL' or 'NON-TECHNICAL'.
#         low_domain:  Sub-discipline filter e.g. 'Instrumentation', 'Civil', 'HSE'.
#                      Takes precedence over high_domain when both are provided.
#     """
#     try:
#         query_embedding = embed_document_large_model(query, dimensions=384)
#         records = neo4j_vector_search_filtered(
#             query_embedding, itb_id,
#             top_k=top_k,
#             high_domain=high_domain,
#             low_domain=low_domain,
#         )

#         formatted = [{
#             "page_chunk":  r.get("current_chunk_text", ""),
#             "file_name":   r.get("document_filename", "N/A"),
#             "page_number": str(r.get("Chunk_SeqNo", "N/A")),
#             "score":       float(r.get("score", 0.0)),
#             "CurrentChunkID": r.get("CurrentChunkID", ""),
#             "low_domain":  r.get("low_level_domain", ""),
#             "high_domain": r.get("high_level_domain", ""),
#             "context_chunks": {
#                 "previous": r.get("prev_chunk_text", ""),
#                 "next":     r.get("next_chunk_text", ""),
#             },
#             "source": "graph_db_filtered",
#         } for r in records]

#         logger.debug(f"neo4j_graph_tool_filtered: {len(formatted)} results")
#         return formatted

#     except Exception as e:
#         logger.error(f"neo4j_graph_tool_filtered failed: {e}")
#         return []


# ---------------------------------------------------------------------------
# Personalized PageRank (PPR) — Graph-Augmented Retrieval
# ---------------------------------------------------------------------------

# PPR_GRAPH_NAME = "itb_retrieval_graph"


# def ensure_gds_projection(tender_id: str | None = None) -> str:
#     """
#     Create the GDS in-memory projection and then mutate it with KNN
#     SIMILAR_TO edges between Chunk nodes.

#     Only NEXT, PART_OF, and REFERS_TO are projected as base relationship types.
#     All relationship types use a uniform 'weight' property so PPR can use
#     a single relationshipWeightProperty across all edge types.

#     Weight semantics:
#       PART_OF    → 1.0  (structural, always full weight)
#       NEXT       → 0.9  (sequential continuity, near-full weight)
#       REFERS_TO  → actual score from graph (0.55–1.0, cross-doc signal)
#       SIMILAR_TO → KNN cosine similarity (0.7–1.0, written by knn.mutate)
#     """
#     graph_name = PPR_GRAPH_NAME

#     exists = neo4j_run_query(
#         "CALL gds.graph.exists($name) YIELD exists RETURN exists",
#         {"name": graph_name},
#     )
#     if exists and exists[0].get("exists"):
#         logger.debug(f"GDS projection '{graph_name}' already exists (with KNN edges)")
#         return graph_name

#     # ── Step 1: Project base graph (Cypher aggregation syntax) ──────────────
#     # Uses CASE to compute a uniform 'weight' property per edge type since
#     # PART_OF and NEXT have no stored properties. REFERS_TO reads its 'score'.
#     project_query = """
#     MATCH (a)-[r]->(b)
#     WHERE type(r) IN ['PART_OF', 'NEXT', 'REFERS_TO']
#       AND (a:Chunk OR a:Document)
#       AND (b:Chunk OR b:Document)
#     WITH a, b, r,
#          CASE type(r)
#            WHEN 'PART_OF'   THEN 1.0
#            WHEN 'NEXT'      THEN 0.9
#            WHEN 'REFERS_TO' THEN coalesce(r.score, 0.55)
#          END AS w
#     WITH gds.graph.project(
#       $name,
#       a,
#       b,
#       {
#         sourceNodeLabels: labels(a),
#         targetNodeLabels: labels(b),
#         sourceNodeProperties: a { .textEmbedding },
#         targetNodeProperties: b { .textEmbedding },
#         relationshipType: type(r),
#         relationshipProperties: { weight: w }
#       },
#       {
#         undirectedRelationshipTypes: ['PART_OF', 'NEXT', 'REFERS_TO']
#       }
#     ) AS proj
#     RETURN proj.graphName AS graphName,
#            proj.nodeCount AS nodeCount,
#            proj.relationshipCount AS relationshipCount
#     """
#     result = neo4j_run_query(project_query, {"name": graph_name})
#     if result:
#         r = result[0]
#         logger.info(
#             f"GDS projection '{r['graphName']}' created: "
#             f"{r['nodeCount']} nodes, {r['relationshipCount']} relationships"
#         )
#     else:
#         logger.error(f"Failed to create GDS projection '{graph_name}'")
#         return graph_name

#     # ── Step 2: Mutate projection with KNN SIMILAR_TO edges ───────────────────
#     # KNN computes cosine similarity between Chunk.textEmbedding vectors and
#     # writes SIMILAR_TO edges into the in-memory projection only (mutate mode).
#     # These edges give PPR direct 1-hop paths between semantically similar
#     # chunks, bypassing the long Document-level traversal paths.
#     #
#     # topK=5       → each chunk connects to its 5 nearest neighbours
#     # cutoff=0.70  → only keep high-confidence similarity edges
#     # weight       → similarity score written as 'weight' (matches projection)
#     knn_query = """
#     CALL gds.knn.mutate(
#       $name,
#       {
#         nodeLabels:            ['Chunk'],
#         nodeProperties:        ['textEmbedding'],
#         topK:                  5,
#         similarityCutoff:      0.70,
#         mutateRelationshipType: 'SIMILAR_TO',
#         mutateProperty:        'weight'
#       }
#     )
#     YIELD relationshipsWritten, similarityDistribution
#     RETURN relationshipsWritten, similarityDistribution
#     """
#     knn_result = neo4j_run_query(knn_query, {"name": graph_name})
#     if knn_result:
#         k = knn_result[0]
#         sim_dist = k.get('similarityDistribution', {})
#         mean_sim = sim_dist.get('mean', 'N/A')
#         logger.info(
#             f"KNN mutate complete: {k['relationshipsWritten']} SIMILAR_TO edges added "
#             f"(mean similarity: {mean_sim if isinstance(mean_sim, str) else f'{mean_sim:.3f}'})"
#         )
#     else:
#         logger.warning("KNN mutate returned no result — PPR will run without SIMILAR_TO edges")

#     return graph_name


# def drop_gds_projection() -> None:
#     """Drop the GDS projection if it exists (e.g. after re-indexing)."""
#     neo4j_run_query(
#         "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName",
#         {"name": PPR_GRAPH_NAME},
#     )
#     logger.info(f"GDS projection '{PPR_GRAPH_NAME}' dropped")


# def resolve_chunk_node_ids(chunk_ids: List[str]) -> List[int]:
#     """Resolve ChunkID strings to internal Neo4j node IDs for GDS sourceNodes."""
#     if not chunk_ids:
#         return []
#     query = """
#     MATCH (c:Chunk)
#     WHERE c.ChunkID IN $chunk_ids
#     RETURN id(c) AS nodeId
#     """
#     results = neo4j_run_query(query, {"chunk_ids": chunk_ids})
#     return [r["nodeId"] for r in results if r.get("nodeId") is not None]


# def fuse_scores_rrf(
#     vector_results: List[Dict[str, Any]],
#     ppr_results: List[Dict[str, Any]],
#     k: int = 60,
#     chunk_id_key: str = "ChunkID",
# ) -> List[Dict[str, Any]]:
#     """
#     Reciprocal Rank Fusion of vector retrieval and PPR graph walk results.

#     Both inputs: list of dicts with at least {chunk_id_key: str}.
#     Returns merged list sorted by RRF score descending.
#     """
#     from collections import defaultdict

#     rrf_scores: Dict[str, float] = defaultdict(float)

#     for rank, item in enumerate(vector_results, start=1):
#         cid = item.get(chunk_id_key)
#         if cid:
#             rrf_scores[cid] += 1.0 / (k + rank)

#     for rank, item in enumerate(ppr_results, start=1):
#         cid = item.get(chunk_id_key)
#         if cid:
#             rrf_scores[cid] += 1.0 / (k + rank)

#     # Merge metadata — PPR results overwrite vector if both present
#     all_chunks: Dict[str, Dict[str, Any]] = {}
#     for item in vector_results:
#         cid = item.get(chunk_id_key)
#         if cid:
#             all_chunks[cid] = item
#     for item in ppr_results:
#         cid = item.get(chunk_id_key)
#         if cid:
#             all_chunks[cid] = item

#     return sorted(
#         [{"rrf_score": score, **all_chunks[cid]} for cid, score in rrf_scores.items()],
#         key=lambda x: x["rrf_score"],
#         reverse=True,
#     )


# ============================================================
# NEW: Document-centric retrieval tools (for document graph)
# ============================================================

@tool
def get_all_documents_for_itb(tender_id: str) -> List[str]:
    """
    Retrieve a list of all document filenames for a given ITB.
    Use when: broad exploration or when user asks for "all documents" or "what documents are available?"
    """
    query = """
    MATCH (i:ITB {ITB_ID: $tender_id})<-[:BELONGS_TO]-(d:Document)
    RETURN d.docURL as filename
    """
    results = neo4j_run_query(query, {"tender_id": tender_id})
    return [r['filename'] for r in results if r.get('filename')]


@tool
async def search_document_summaries(keyword: str, tender_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search document summaries and metadata for keywords.
    Use when: User asks general questions that might match document content/titles.
    """
    query = """
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    WHERE d.summary CONTAINS $keyword OR d.docURL CONTAINS $keyword
    RETURN 
      d.docURL as filename,
      d.summary as summary_text,
      d.author as author,
      d.creation_date as creation_date
    LIMIT $top_k
    """
    results = neo4j_run_query(query, {"tender_id": tender_id, "keyword": keyword, "top_k": top_k})
    logger.info(f"search_document_summaries: found {len(results)} documents for keyword '{keyword}'")
    return results


def _get_all_entities(tender_id: str) -> Dict[str, List[str]]:
    """
    Fetch all entity names from the graph for a given ITB, grouped by type.
    Returns: {"Identity": [...], "Place": [...], "Topic": [...]}
    """
    query = """
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[:MENTION]->(e:Identity)
    WHERE e.identityName IS NOT NULL
    RETURN DISTINCT 'Identity' AS type, e.identityName AS name
    UNION
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[:LOCATED]->(e:Place)
    WHERE e.placeName IS NOT NULL
    RETURN DISTINCT 'Place' AS type, e.placeName AS name
    UNION
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[:HAS_TOPIC]->(e:Topic)
    WHERE e.topicName IS NOT NULL
    RETURN DISTINCT 'Topic' AS type, e.topicName AS name
    """
    records = neo4j_run_query(query, {"tender_id": tender_id})
    grouped: Dict[str, List[str]] = {"Identity": [], "Place": [], "Topic": []}
    for r in records:
        t = r.get("type")
        n = r.get("name")
        if t in grouped and n:
            grouped[t].append(n)
    return grouped


def _fuzzy_resolve_entity(entity_name: str, candidates: List[str], cutoff: float = 0.6) -> Optional[str]:
    """
    Return the best fuzzy match from candidates for entity_name, or None.
    Uses difflib sequence matching (handles abbreviations, partial names, typos).
    cutoff: minimum similarity ratio (0–1). 0.6 is a reasonable default.
    """
    import difflib
    if not candidates:
        return None

    lower = entity_name.lower()
    for c in candidates:
        if lower in c.lower() or c.lower() in lower:
            return c
    matches = difflib.get_close_matches(entity_name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


@tool
def find_documents_by_entity(entity_type: str, entity_name: str, tender_id: str) -> List[Dict[str, Any]]:
    """
    Find documents mentioning a specific entity (company, place, topic).
    entity_type: 'Identity', 'Place', 'Topic'
    Use when: User mentions companies, locations, standards, topics etc.
    Performs fuzzy matching against known graph entities before querying.
    """
    # ── 1. Load all graph entities and fuzzy-resolve the input name ──────────
    all_entities = _get_all_entities(tender_id)

    # Search across all types unless a specific type is given
    search_types = (
        [entity_type] if entity_type in all_entities
        else list(all_entities.keys())
    )

    resolved_name = entity_name  
    resolved_type = None

    for etype in search_types:
        match = _fuzzy_resolve_entity(entity_name, all_entities[etype])
        if match:
            resolved_name = match
            resolved_type = etype
            logger.info(
                f"find_documents_by_entity: fuzzy resolved "
                f"'{entity_name}' → '{resolved_name}' ({resolved_type})"
            )
            break

    if resolved_type is None:
        logger.warning(
            f"find_documents_by_entity: no fuzzy match for '{entity_name}' "
            f"in {list(all_entities.keys())} — falling back to CONTAINS search"
        )

    # ── 2. Query — exact match on resolved name, or CONTAINS as fallback ────
    if resolved_type:
        # Exact match on the resolved (graph-canonical) name
        query = """
        MATCH (e)
        WHERE (e:Identity AND e.identityName = $entity_name)
           OR (e:Place    AND e.placeName    = $entity_name)
           OR (e:Topic    AND e.topicName    = $entity_name)
        MATCH (e)<-[rel]-(d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
        RETURN
          d.docURL as filename,
          CASE
            WHEN e.identityName IS NOT NULL THEN e.identityName
            WHEN e.placeName    IS NOT NULL THEN e.placeName
            WHEN e.topicName    IS NOT NULL THEN e.topicName
          END as entity,
          d.summary as summary_text,
          type(rel) as relationship_type
        LIMIT 10
        """
    else:
        # Fallback: case-insensitive substring scan
        query = """
        MATCH (e)
        WHERE (e:Identity AND toLower(e.identityName) CONTAINS toLower($entity_name))
           OR (e:Place    AND toLower(e.placeName)    CONTAINS toLower($entity_name))
           OR (e:Topic    AND toLower(e.topicName)    CONTAINS toLower($entity_name))
        MATCH (e)<-[rel]-(d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
        RETURN
          d.docURL as filename,
          CASE
            WHEN e.identityName IS NOT NULL THEN e.identityName
            WHEN e.placeName    IS NOT NULL THEN e.placeName
            WHEN e.topicName    IS NOT NULL THEN e.topicName
          END as entity,
          d.summary as summary_text,
          type(rel) as relationship_type
        LIMIT 10
        """

    results = neo4j_run_query(query, {"entity_name": resolved_name, "tender_id": tender_id})
    logger.info(f"find_documents_by_entity: found {len(results)} documents for entity '{resolved_name}'")
    return results


def _get_all_domains(tender_id: str) -> Dict[str, List[str]]:
    """
    Fetch all domain names from the graph for a given ITB, grouped by level.
    Returns: {"high": [...], "mid": [...], "low": [...]}
    """
    query = """
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[:HAS_HIGH_DOMAIN]->(hd:HighLevelDomain)
    WHERE hd.domainName IS NOT NULL
    RETURN DISTINCT 'high' AS level, hd.domainName AS name
    UNION
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[:HAS_MID_LEVEL]->(md:MidLevelDomain)
    WHERE md.domainName IS NOT NULL
    RETURN DISTINCT 'mid' AS level, md.domainName AS name
    UNION
    MATCH (d:Document)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    MATCH (d)-[:HAS_LOW_LEVEL]->(ld:LowLevelDomain)
    WHERE ld.domainName IS NOT NULL
    RETURN DISTINCT 'low' AS level, ld.domainName AS name
    """
    records = neo4j_run_query(query, {"tender_id": tender_id})
    grouped: Dict[str, List[str]] = {"high": [], "mid": [], "low": []}
    for r in records:
        lv = r.get("level")
        nm = r.get("name")
        if lv in grouped and nm:
            grouped[lv].append(nm)
    return grouped


@tool
def find_documents_by_domain(domain_name: str, tender_id: str, level: str = "low") -> List[Dict[str, Any]]:
    """
    Find documents within a specific domain by level (high, mid, or low).
    level: 'high', 'mid', or 'low'
    Use when: User asks about specific domains like 'Safety', 'Commercial', 'Technical', etc.
    Performs fuzzy matching against known graph domains before querying.
    """
    # ── 1. Load all graph domains and fuzzy-resolve the input name ───────────
    all_domains = _get_all_domains(tender_id)

    resolved_name = domain_name
    resolved_level = level

    # Search the hinted level first, then fall back to all levels
    levels_to_search = (
        [level] + [lv for lv in ("high", "mid", "low") if lv != level]
    )

    for lv in levels_to_search:
        match = _fuzzy_resolve_entity(domain_name, all_domains[lv])
        if match:
            resolved_name = match
            resolved_level = lv
            logger.info(
                f"find_documents_by_domain: fuzzy resolved "
                f"'{domain_name}' → '{resolved_name}' (level={resolved_level})"
            )
            break
    else:
        logger.warning(
            f"find_documents_by_domain: no fuzzy match for '{domain_name}' "
            f"across all levels — using original name at level='{level}'"
        )

    # ── 2. Build query with resolved level ───────────────────────────────────
    if resolved_level == "high":
        domain_node = "HighLevelDomain"
        relation = "HAS_HIGH_DOMAIN"
    elif resolved_level == "mid":
        domain_node = "MidLevelDomain"
        relation = "HAS_MID_LEVEL"
    else:
        domain_node = "LowLevelDomain"
        relation = "HAS_LOW_LEVEL"

    query = f"""
    MATCH (d:Document)-[:{relation}]->(domain:{domain_node})
    MATCH (d)-[:BELONGS_TO]->(i:ITB {{ITB_ID: $tender_id}})
    WHERE domain.domainName = $domain_name
    RETURN 
      d.docURL as filename,
      d.summary as summary_text,
      domain.domainDescription as domain_context,
      d.creation_date as creation_date
    LIMIT 10
    """
    results = neo4j_run_query(query, {"domain_name": resolved_name, "tender_id": tender_id})
    logger.info(f"find_documents_by_domain: found {len(results)} documents in domain '{resolved_name}' (level={resolved_level})")
    return results


@tool
def find_related_documents(filename: str, tender_id: str) -> List[Dict[str, Any]]:
    """
    Find documents related to a given document via REFERS_TO relationships.
    Sorted by composite relevance score (semantic + reference + domain).
    Use when: You found a relevant document and want to find similar or referenced ones.
    """
    query = """
    MATCH (source:Document {docURL: $filename})-[r:REFERS_TO]->(target:Document)
    MATCH (target)-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    RETURN 
      target.docURL as filename,
      target.summary as summary_text,
      r.score as relevance_score,
      r.semantic_score as semantic_score,
      r.reference_score as reference_score,
      r.domain_score as domain_score,
    ORDER BY r.score DESC
    LIMIT 5
    """
    results = neo4j_run_query(query, {"filename": filename, "tender_id": tender_id})
    logger.info(f"find_related_documents: found {len(results)} related documents for '{filename}'")
    return results


@tool
def get_document_details(filename: str, tender_id: str) -> Dict[str, Any]:
    """
    Get full document properties including metadata, domains, and associated entities.
    Use when: You need to enrich a document with all its context and relationships.
    """
    query = """
    MATCH (d:Document {docURL: $filename})-[:BELONGS_TO]->(i:ITB {ITB_ID: $tender_id})
    OPTIONAL MATCH (d)-[:HAS_HIGH_DOMAIN]->(hd:HighLevelDomain)
    OPTIONAL MATCH (d)-[:HAS_MID_LEVEL]->(md:MidLevelDomain)
    OPTIONAL MATCH (d)-[:HAS_LOW_LEVEL]->(ld:LowLevelDomain)
    OPTIONAL MATCH (d)-[:MENTION]->(entity:Identity)
    OPTIONAL MATCH (d)-[:LOCATED]->(place:Place)
    OPTIONAL MATCH (d)-[:HAS_TOPIC]->(topic:Topic)
    
    RETURN 
      d.docURL as filename,
      d.summary as summary,
      d.creation_date as creation_date,
      d.lastModifDate as last_modified,
      d.referencedDocs as referenced_docs,
      collect(DISTINCT hd.domainName) as high_domains,
      collect(DISTINCT md.domainName) as mid_domains,
      collect(DISTINCT ld.domainName) as low_domains,
      collect(DISTINCT entity.identityName) as entities,
      collect(DISTINCT place.placeName) as locations,
      collect(DISTINCT topic.topicName) as topics,
    """
    results = neo4j_run_query(query, {"filename": filename, "tender_id": tender_id})
    if results:
        logger.info(f"get_document_details: retrieved details for '{filename}'")
        return results[0]
    logger.warning(f"get_document_details: no document found for '{filename}'")
    return {}


def close_neo4j_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        logger.info("Neo4j Bolt connection closed.")
        
        
        