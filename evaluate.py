import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, FiltrationConfig
from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_groq import ChatGroq

from backend.llm import GROQ_TOOL_RETRY
from backend.settings import JUDGE_MODEL
from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, base_embeddings

load_dotenv()

PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")
MAX_CONTEXTS        = 2
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7


class GroqJudge(DeepEvalBaseLLM):
    """Groq (free tier) as the deepeval judge instead of OpenAI."""

    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        super().__init__(model)

    def load_model(self):
        return ChatGroq(model=self.model_name, temperature=0, max_retries=6)

    def _structured(self, schema):
        return self.model.with_structured_output(schema, method="json_schema").with_retry(**GROQ_TOOL_RETRY)

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self._structured(schema).invoke(prompt)
        return self.model.invoke(prompt).text

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self._structured(schema).ainvoke(prompt)
        return (await self.model.ainvoke(prompt)).text

    def get_model_name(self) -> str:
        return f"groq/{self.model_name}"


class CohereEmbedder(DeepEvalBaseEmbeddingModel):
    """Same Cohere embeddings as the app, for golden generation instead of OpenAI."""

    def __init__(self):
        super().__init__(base_embeddings.model)

    def load_model(self):
        return base_embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.model.aembed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.model.aembed_documents(texts)

    def get_model_name(self) -> str:
        return f"cohere/{base_embeddings.model}"


def generate_goldens() -> list[dict]:
    judge = GroqJudge()
    synthesizer = Synthesizer(
        model=judge,
        max_concurrent=2,
        filtration_config=FiltrationConfig(critic_model=judge),
    )
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
            critic_model=judge,
            embedder=CohereEmbedder(),
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    if not pairs:
        # deepeval swallows judge errors (e.g. Groq daily token limit) and returns nothing.
        sys.exit("No goldens generated (judge calls likely failed / rate-limited). goldens.json not written.")
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def main() -> None:
    pairs = load_goldens() if GOLDENS_FILE.exists() else []
    if not pairs:
        pairs = generate_goldens()

    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")

    judge = GroqJudge()
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge),
    ]

    test_cases = []
    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] RAG query: {pair['input'][:70]}...", flush=True)
        session_id = f"evaluation_session_{uuid4()}"
        add_paper(docs, session_id)

        query = pair["input"] + " as per the report in knowledge base"
        answer, retrieval_context = run_rag_query(graph, query, session_id)
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    print("\nScoring with deepeval (Groq judge)...", flush=True)
    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(max_concurrent=1, throttle_value=10),
    )

    summary = []
    for test_result in results.test_results:
        summary.append({
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    if not summary:
        sys.exit("No test results (judge calls likely failed / rate-limited). eval_results.json not overwritten.")

    results_path = Path("eval_results.json")
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_path}.")


if __name__ == "__main__":
    main()
