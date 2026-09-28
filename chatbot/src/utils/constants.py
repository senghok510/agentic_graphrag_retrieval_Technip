"""
Constants and configuration values.

Centralizes magic numbers, thresholds, and keyword mappings used throughout the application.
"""

# Azure Search Configuration
SEARCH_TOP_K = 20
"""Number of top results to retrieve from Azure Search"""

VECTOR_K_NEAREST_NEIGHBORS = 20
"""Number of nearest neighbors for vector search"""

# Format Detection Keywords
FORMAT_KEYWORDS = {
    "table": ["table", "tabular", "matrix", "comparison chart"],
    "list": ["list", "checklist", "bullet points", "bullet", "itemize", "enumerate"],
    "schedule": ["schedule", "timeline", "phases", "milestones"],
    "summary": ["summary", "overview", "executive summary"],
}
"""
Keyword mappings for detecting user intent to format responses.

When a user query contains any of these keywords, the appropriate
formatting agent is triggered to structure the response accordingly.
"""

# Fact Checking Support Levels
SUPPORT_LEVEL_SUPPORTED = "supported"
"""Documents fully support the user's query with clear evidence"""

SUPPORT_LEVEL_PARTIALLY_SUPPORTED = "partially_supported"
"""Documents provide some relevant information but don't fully answer the query"""

SUPPORT_LEVEL_NOT_SUPPORTED = "not_supported"
"""Documents don't provide relevant information to support the query"""

VALID_SUPPORT_LEVELS = [
    SUPPORT_LEVEL_SUPPORTED,
    SUPPORT_LEVEL_PARTIALLY_SUPPORTED,
    SUPPORT_LEVEL_NOT_SUPPORTED,
]
"""All valid support levels for fact checking"""

# Confidence Thresholds
LOW_CONFIDENCE_THRESHOLD = 0.0
"""Threshold below which responses are considered low confidence"""

# Response Configuration
RESPONSE_EMOJIS = ["📝", "📊", "✅", "📌", "💡"]
"""Emojis used to highlight sections in responses"""

# Vector Embedding Field
VECTOR_EMBEDDING_FIELD = "page_embedding"
"""Field name for vector embeddings in Azure Search index"""
