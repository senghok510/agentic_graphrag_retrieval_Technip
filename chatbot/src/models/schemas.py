from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class HelloRequest(BaseModel):
    name: str
class HelloResponse(BaseModel):
    message: str
    
class ReferenceGeneration(BaseModel):
    """Reference to a source document."""
    file_name: str = Field(..., description="The name of the source file.")
    page_number: str = Field(..., description="The page number within the source file.")


class ResponseGeneration(BaseModel):
    """Final response with answer, references, and confidence score."""
    answer: str = Field(..., description="The final answer to the user's question.")
    references: List[ReferenceGeneration] = Field(..., description="List of references used to generate the answer.")
    confidence_score: Optional[float] = Field(default=0.0, description="Confidence score for the answer")
    trace_id: Optional[str] = Field(default=None, description="Unique trace ID for end-to-end traceability")
    log_id: Optional[str] = Field(default=None, description="Unique log ID for linking feedback")


class FactCheckerOutput(BaseModel):
    """Output from the fact checker agent."""
    answer: str = Field(..., description="The clean, corrected final answer to show to the user — NO audit commentary, NO sentence-by-sentence breakdown, just the polished answer text with inline [Source X] citations.")
    sources: List[str] = Field(default=[], description="List of sources used to generate the answer.")
    is_supported: Literal["supported", "partially_supported", "not_supported"] = Field(
        ...,
        description="Support status for the answer"
    )
    note: Optional[str] = Field(default=None, description="Additional notes about the answer")


class ComplexityAnalysis(BaseModel):
    """Analysis of query complexity for multi-hop reasoning."""
    reasoning_steps: str = Field(description="Step-by-step analysis of why this query is complex or simple")
    is_complex: bool = Field(description="Whether the query requires multi-hop reasoning")
    reasoning: str = Field(description="Final explanation of complexity assessment")
    sub_questions: List[str] = Field(default=[], description="Sub-questions if complex")







class SimpleQueryExpansion(BaseModel):
    """Structured output for simple query expansion."""

    rephrased_questions: List[str] = Field(
        default=[],
        description=(
            "2 rephrased versions of the original query focusing on: technical requirements, legal/contractual obligations, commercial/vendor implications"
        ),
    )
    keywords: List[str] = Field(
        description="3-5 keywords covering possible interpretations of the query within the ITB/EPC domain"
    )

class ComplexQueryExpansion(BaseModel):
    """Structured output for complex query expansion."""
    rephrased_questions: List[str] = Field(
        default=[],
        description="3 rephrased versions of the original query focusing on: technical requirements, legal/contractual obligations, commercial/vendor implications"
    )
    keywords: List[str] = Field(
        description="6-8 keywords covering the technical, legal, and commercial dimensions"
    )


class RoutingDecision(BaseModel):
    """Routing decision for response generation."""
    route: str = Field(description="Route to take: low_confidence, format, short_respond, or detailed_respond")
    reasoning: str = Field(description="Explanation for routing decision")
    expected_length: str = Field(description="Expected response length: brief, moderate, or comprehensive")

class FollowupQuestionIdentifier(BaseModel):
    """ Decides if the current user query is new or a followup question"""
    query: str = Field(description="Reformulated user query with resolved context from chat history if needed")
    is_follow_up: bool = Field(description="True if query is a follow-up on past conversation, False if it's a standalone new question")


class ErrorClassification(BaseModel):
    """Classification of errors in RAG pipeline for granular feedback."""
    retriever_errors: List[str] = Field(
        default=[],
        description="List of retriever error tags: Missing Document, Off-topic Document, Truncated Context, Poor Ranking, Outdated Source"
    )
    generator_errors: List[str] = Field(
        default=[],
        description="List of generator error tags: Hallucination, Incorrect Format, Faulty Reasoning, Contradicts Sources, Lacks Precision"
    )
    flagged_sources: List[str] = Field(
        default=[],
        description="List of source IDs flagged as irrelevant or incorrect"
    )


class SummaryResponse(BaseModel):
    """Structured project summary response with key project details."""
    project_name: str = Field(..., description="Project Name. If not found, use 'N/A'.")
    scope: str = Field(..., description="Overall Scope. If not found, use 'N/A'.")
    location: str = Field(..., description="Project Location. If not found, use 'N/A'.")
    contract_type: str = Field(..., description="Contract Type. If not found, use 'N/A'.")
    project_currency: str = Field(..., description="Project Currency. If not found, use 'N/A'.")
    itb_receipt_date: str = Field(..., description="ITB Receipt Date. If not found, use 'N/A'.")
    commercial_bid_due_date: str = Field(..., description="Commercial bid due date. If not found, use 'N/A'.")
    technical_bid_due_date: str = Field(..., description="Technical bid due date. If not found, use 'N/A'.")
    expected_award_date: str = Field(..., description="Expected award date. If not found, use 'N/A'.")
    itb_phase_timeline: str = Field(
        ...,
        description="Timeline for the ITB phase, including key dates and events. Format as a clear descriptive string or list of events. If not found, use 'N/A'."
    )
    # Answers to specific questions
    project_scope_of_work: str = Field(..., description="Detailed summary of project scope of work. If not found, use 'N/A'.")
    project_schedule: str = Field(..., description="Detailed summary of project schedule. If not found, use 'N/A'.")
    
    references: List[ReferenceGeneration] = Field(default=[], description="Sources used")
    confidence_score: float = Field(default=0.0, description="Confidence score for the summary extraction")


class DisciplineSummaryRequest(BaseModel):
    """Request for discipline-specific project summary."""
    tender_id: str = Field(..., description="ITB identifier")
    discipline: str = Field(..., description="Discipline name (e.g., Process, Mechanical)")


class ChecklistItemResponse(BaseModel):
    """Result for a single checklist question."""
    question: str = Field(..., description="The checklist question")
    answer: str = Field(..., description="The answer found in documents")
    references: List[ReferenceGeneration] = Field(default=[], description="Sources used for this answer")


class DisciplineSummaryResponse(BaseModel):
    """Structured summary for a specific discipline."""
    discipline: str = Field(..., description="The discipline name")
    results: List[ChecklistItemResponse] = Field(..., description="Answers to checklist questions")
    overall_summary: str = Field(..., description="A synthesized summary for the discipline")
    confidence_score: float = Field(default=0.0, description="Overall confidence score")


class MissingDocument(BaseModel):
    """Details about a referenced document that is missing from the system."""
    filename: str = Field(..., description="Name of the missing document")
    referenced_in: str = Field(..., description="Where the missing document was mentioned")
    reasoning: str = Field(..., description="Context of why it's considered missing or required")


class Contradiction(BaseModel):
    """Details about conflicting information found across documents."""
    topic: str = Field(..., description="Subject of the contradiction (e.g., Schedule, LDs)")
    finding: str = Field(..., description="Description of the conflict")
    document_a: str = Field(..., description="First document providing conflicting info")
    document_b: str = Field(..., description="Second document providing conflicting info")
    impact: str = Field(..., description="Potential impact of this contradiction")


class GapAnalysisReport(BaseModel):
    """Final report for project gap analysis."""
    tender_id: str = Field(..., description="ITB identifier")
    missing_documents: List[MissingDocument] = Field(default=[], description="List of identified missing documents")
    contradictions: List[Contradiction] = Field(default=[], description="List of identified contradictions")
    summary: str = Field(..., description="Cohesive executive summary of the gap analysis")
    confidence_score: float = Field(default=0.0, description="Overall confidence in the analysis")