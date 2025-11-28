from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

INVESTIGATION_SYSTEM = """You are an expert financial crime investigator. 
Analyze the flagged transaction and related case history to determine:
1. Risk level (LOW/MEDIUM/HIGH/CRITICAL)
2. Pattern matches with known fraud typologies
3. Recommended action (APPROVE/REVIEW/ESCALATE/BLOCK)
4. Key evidence supporting your assessment
Be concise and specific. Reference precedent cases when relevant."""


class CaseState(TypedDict):
    transaction: Dict[str, Any]
    risk_score: float
    precedent_cases: List[str]
    analysis: str
    recommendation: str
    confidence: float
    session_id: str


class InvestigationAgent:
    def __init__(self, opensearch_endpoint: str, openai_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=openai_model, temperature=0.1)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.opensearch_endpoint = opensearch_endpoint
        self._build_graph()

    def _retrieve_precedents(self, state: CaseState) -> CaseState:
        try:
            vectorstore = OpenSearchVectorSearch(
                index_name="fraud-cases",
                embedding_function=self.embeddings,
                opensearch_url=self.opensearch_endpoint
            )
            query = f"fraud transaction amount {state['transaction'].get('amount')} pattern"
            docs = vectorstore.similarity_search(query, k=5)
            state["precedent_cases"] = [d.page_content for d in docs]
        except Exception as e:
            logger.warning(f"Precedent retrieval failed: {e}")
            state["precedent_cases"] = []
        return state

    def _analyze_case(self, state: CaseState) -> CaseState:
        precedents = "\n".join(state["precedent_cases"][:3]) or "No precedent cases found"
        messages = [
            SystemMessage(content=INVESTIGATION_SYSTEM),
            HumanMessage(content=f"""
Transaction: {state["transaction"]}
Risk Score: {state["risk_score"]:.3f}
Precedent Cases:\n{precedents}
Provide your investigation analysis and recommendation.
""")
        ]
        response = self.llm.invoke(messages)
        state["analysis"] = response.content
        if "CRITICAL" in response.content: state["recommendation"] = "BLOCK"
        elif "ESCALATE" in response.content: state["recommendation"] = "ESCALATE"
        elif "REVIEW" in response.content: state["recommendation"] = "REVIEW"
        else: state["recommendation"] = "APPROVE"
        state["confidence"] = min(0.95, 0.6 + state["risk_score"] * 0.35)
        return state

    def _build_graph(self):
        workflow = StateGraph(CaseState)
        workflow.add_node("retrieve", self._retrieve_precedents)
        workflow.add_node("analyze", self._analyze_case)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", END)
        self.graph = workflow.compile()

    def investigate(self, transaction: dict, risk_score: float, session_id: str) -> CaseState:
        initial = {"transaction": transaction, "risk_score": risk_score,
                   "precedent_cases": [], "analysis": "", "recommendation": "",
                   "confidence": 0.0, "session_id": session_id}
        return self.graph.invoke(initial)
