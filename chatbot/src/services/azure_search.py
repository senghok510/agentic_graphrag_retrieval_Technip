import logging
from typing import Any

from azure.search.documents.models import QueryType, VectorizedQuery
from langchain_core.tools import tool

from ..config.appSettings import ai_search_client, embed_document_large_model, get_settings
from ..utils.constants import SEARCH_TOP_K, VECTOR_EMBEDDING_FIELD, VECTOR_K_NEAREST_NEIGHBORS

logger = logging.getLogger("agent_flow.azure_search")


def get_search_index() -> str:
    """
    Return the appropriate Azure Search index based on Tender_ID.

    Args:
        tender_id: The ITB identifier

    Returns:
        str: The name of the Azure Search index to use
    """
    selected_index = get_settings().AZURE_SEARCH_INDEX_NAME
    logger.debug(f"Using unified APIM index: {selected_index}")
    return selected_index


### vector_query_text: hyde_doc,
@tool
def azure_search_tool(
    query: str,
    tender_id: str | None = None,
    top: int = SEARCH_TOP_K,
    vector_query_text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Query Azure Search index and return sorted results by reranker_score.

    This tool performs hybrid search (semantic + vector) on the specified Azure Search index.
    Results include page_chunk, file_name, and page_number.

    Args:
        query: Search query text (used for semantic search)
        tender_id: Tender identifier to select the appropriate search index
        top: Maximum number of results to return
        vector_query_text: Optional text to use for vector embeddings (if different from query)

    Returns:
        List of dictionaries containing page_chunk, file_name, and page_number
    """
    try:
        # vector_query_text = hydoc
        # query = expanded query
        text_to_embed = vector_query_text or query
        search_client = ai_search_client()

        # Create vector query using embeddings
        # K most similar chunks to the hydoc vectors
        vector_query = VectorizedQuery(
            vector=embed_document_large_model(text_to_embed, dimensions=3072),
            k_nearest_neighbors=VECTOR_K_NEAREST_NEIGHBORS,
            fields=VECTOR_EMBEDDING_FIELD,
        )

        # Determine the tender_id filter
        filter_expr = None
        if tender_id:
            filter_expr = f"tender_id eq '{tender_id}'"
        logger.info(f"Received query = {query}")
        logger.debug(f"DEBUG azure_search_tool: received query={query[:150] if query else 'NONE'}")
        logger.info(
            f"Azure Search Config - Query: '{query}', ITB: '{tender_id}', Filter: '{filter_expr}'"
        )

        # Perform hybrid search (semantic + vector)
        try:
            logger.debug(
                f"Attempting Semantic Hybrid Search on index {get_settings().AZURE_SEARCH_INDEX_NAME}"
            )
            semantic_config = get_settings().AI_SEARCH_SEMANTIC_SEARCH_CONFIG
            if not semantic_config:
                raise Exception(
                    "SemanticQueriesNotAvailable: AI_SEARCH_SEMANTIC_SEARCH_CONFIG is not set"
                )
            logger.debug(
                f"Attempting Semantic Hybrid Search on index {get_settings().AZURE_SEARCH_INDEX_NAME}"
            )
            results = search_client.search(
                search_text=query,
                query_type=QueryType.SEMANTIC,
                vector_queries=[vector_query],
                top=top,
                filter=filter_expr,
                include_total_count=True,
                semantic_configuration_name=semantic_config,
            )
            logger.debug("Semantic Hybrid Search executed, collecting results...")

            # Collect all fields for sorting
            result_list_intermediate = [
                {
                    "page_chunk": result["page_chunk"],
                    "file_name": result["file_name"],
                    "page_number": result["page_number"],
                    "CurrentChunkID": result.get("chunk_id", ""),
                    "reranker_score": result.get(
                        "@search.reranker_score", result.get("@search.score", 0)
                    ),
                }
                for result in results
            ]
        except Exception as semantic_err:
            if "Semantic search is not enabled" in str(
                semantic_err
            ) or "SemanticQueriesNotAvailable" in str(semantic_err):
                logger.warning(
                    f"Semantic search not available, falling back to standard hybrid search: {semantic_err}"
                )
                # Fallback to standard vector + keyword search without Semantic ranking
                results = search_client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=top,
                    filter=filter_expr,
                    include_total_count=True,
                )

                # Collect all fields for sorting using standard search score
                result_list_intermediate = [
                    {
                        "page_chunk": result["page_chunk"],
                        "file_name": result["file_name"],
                        "page_number": result["page_number"],
                        "CurrentChunkID": result.get("chunk_id", ""),
                        "reranker_score": result.get("@search.score", 0),
                    }
                    for result in results
                ]
            else:
                raise semantic_err

        # Log score statistics only
        if result_list_intermediate:
            scores = [r["reranker_score"] for r in result_list_intermediate]
            logger.debug(
                f"Azure Search: {len(result_list_intermediate)} results, scores {min(scores):.3f}-{max(scores):.3f}"
            )
        else:
            logger.warning("Azure Search: 0 results")

        # Sort by reranker_score descending
        result_list_intermediate.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Return with score and source for hybrid retrieval
        result_list = [
            {
                "page_chunk": result["page_chunk"],
                "file_name": result["file_name"],
                "page_number": result["page_number"],
                "CurrentChunkID": result["CurrentChunkID"],
                "reranker_score": result["reranker_score"],
                "source": "vector_db",
            }
            for result in result_list_intermediate
        ]

        return result_list

    except Exception as e:
        logger.error(f"Azure Search error: {e}")
        return []
