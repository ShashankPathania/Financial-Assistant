"""
Local LLM-as-a-Judge Evaluator
================================
Uses a local Ollama instance (llama3.1:8b) for offline, token-heavy RAG
evaluation.  Calculates Precision@K, Recall@K, Faithfulness, Relevance,
Completeness, and Clarity.  Auto-generates Q/A pairs for testing.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ollama as ollama_client
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LocalLLMEvaluator:
    """
    Offline RAG evaluator using a local Ollama LLM instance.

    Ports the EnterpriseRAGEvaluator from the baseline, but routes all
    LLM calls to Ollama instead of Groq to conserve API limits.
    """

    EVAL_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system 
designed for financial document analysis.

Evaluate the following RAG output on these criteria. For each criterion, provide:
1. A score from 1 to 5 (1=poor, 5=excellent)
2. A brief justification (1-2 sentences)

CRITERIA:
- **Faithfulness**: Does the answer stick to the provided context? Are there any hallucinated facts?
- **Relevance**: Does the answer address the user's question directly?
- **Completeness**: Does the answer cover all aspects of the question using available context?
- **Clarity**: Is the answer well-structured, clear, and professional?

USER QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Respond with ONLY a JSON object (no markdown, no extra text):
{{
    "faithfulness": {{"score": <1-5>, "justification": "..."}},
    "relevance": {{"score": <1-5>, "justification": "..."}},
    "completeness": {{"score": <1-5>, "justification": "..."}},
    "clarity": {{"score": <1-5>, "justification": "..."}}
}}"""

    QA_GENERATION_PROMPT = """You are a financial analyst creating test questions for a document Q&A system.

Given the following document excerpt, generate exactly {count} question-answer pairs that test 
different aspects of the content. Focus on specific facts, numbers, and concepts.

DOCUMENT EXCERPT:
{document_text}

Respond with ONLY a JSON array (no markdown):
[
    {{"question": "...", "expected_answer": "..."}},
    ...
]"""

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.evaluation_results: List[Dict[str, Any]] = []

        logger.info("LocalLLMEvaluator initialized (model=%s, base_url=%s)", self.model, self.base_url)

    # ------------------------------------------------------------------ #
    #  Ollama LLM Call
    # ------------------------------------------------------------------ #
    def _call_ollama(self, prompt: str, temperature: float = 0.3) -> str:
        """Call the local Ollama instance."""
        try:
            response = ollama_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 1000,
                },
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            return ""

    # ------------------------------------------------------------------ #
    #  Per-Sample Evaluation
    # ------------------------------------------------------------------ #
    def evaluate_sample(
        self,
        question: str,
        context: str,
        answer: str,
        expected_answer: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG output sample.

        Returns:
            Dict with scores for faithfulness, relevance, completeness, clarity,
            plus a weighted_average score.
        """
        prompt = self.EVAL_PROMPT.format(
            question=question,
            context=context[:3000],  # truncate for context window
            answer=answer,
        )

        logger.info("Evaluating sample: '%s'", question[:80])
        raw_response = self._call_ollama(prompt, temperature=0.1)

        if not raw_response:
            logger.error("Empty response from Ollama for evaluation.")
            return self._default_scores(question)

        try:
            # Clean up potential markdown
            text = raw_response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            scores = json.loads(text)

            # Extract numeric scores
            result = {
                "question": question,
                "faithfulness": scores.get("faithfulness", {}).get("score", 0),
                "faithfulness_reason": scores.get("faithfulness", {}).get("justification", ""),
                "relevance": scores.get("relevance", {}).get("score", 0),
                "relevance_reason": scores.get("relevance", {}).get("justification", ""),
                "completeness": scores.get("completeness", {}).get("score", 0),
                "completeness_reason": scores.get("completeness", {}).get("justification", ""),
                "clarity": scores.get("clarity", {}).get("score", 0),
                "clarity_reason": scores.get("clarity", {}).get("justification", ""),
            }

            # Weighted average (faithfulness weighted highest to penalize hallucinations)
            weights = {"faithfulness": 0.35, "relevance": 0.25, "completeness": 0.20, "clarity": 0.20}
            result["weighted_average"] = round(
                sum(result[k] * w for k, w in weights.items()), 2
            )

            logger.info("Evaluation scores: F=%.1f R=%.1f C=%.1f Cl=%.1f (avg=%.2f)",
                        result["faithfulness"], result["relevance"],
                        result["completeness"], result["clarity"],
                        result["weighted_average"])
            return result

        except json.JSONDecodeError as e:
            logger.error("Failed to parse evaluation JSON: %s\nRaw: %s", e, raw_response[:500])
            return self._default_scores(question)

    def _default_scores(self, question: str) -> Dict[str, Any]:
        """Return default scores when evaluation fails."""
        return {
            "question": question,
            "faithfulness": 0, "faithfulness_reason": "Evaluation failed",
            "relevance": 0, "relevance_reason": "Evaluation failed",
            "completeness": 0, "completeness_reason": "Evaluation failed",
            "clarity": 0, "clarity_reason": "Evaluation failed",
            "weighted_average": 0,
        }

    # ------------------------------------------------------------------ #
    #  Retrieval Metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5,
    ) -> float:
        """Calculate Precision@K."""
        if not retrieved_ids or k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for rid in top_k if rid in relevant_set)
        return round(hits / min(k, len(top_k)), 4)

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5,
    ) -> float:
        """Calculate Recall@K."""
        if not relevant_ids or k <= 0:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(top_k & relevant_set)
        return round(hits / len(relevant_set), 4)

    # ------------------------------------------------------------------ #
    #  Auto-Generate Q/A Pairs
    # ------------------------------------------------------------------ #
    def generate_qa_pairs(
        self,
        document_text: str,
        count: int = 5,
    ) -> List[Dict[str, str]]:
        """Auto-generate Q/A pairs from document text using Ollama."""
        prompt = self.QA_GENERATION_PROMPT.format(
            document_text=document_text[:4000],
            count=count,
        )

        logger.info("Generating %d Q/A pairs from document text.", count)
        raw = self._call_ollama(prompt, temperature=0.5)

        if not raw:
            logger.error("Failed to generate Q/A pairs.")
            return []

        try:
            text = raw
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            qa_pairs = json.loads(text)
            logger.info("Generated %d Q/A pairs.", len(qa_pairs))
            return qa_pairs[:count]
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Q/A pairs JSON: %s", e)
            return []

    # ------------------------------------------------------------------ #
    #  Full Evaluation Run
    # ------------------------------------------------------------------ #
    def run_evaluation(
        self,
        pipeline_fn,
        document_text: str,
        num_samples: int = 5,
        output_path: str = "data/evaluation_report.json",
    ) -> Dict[str, Any]:
        """
        Run a full evaluation:
          1. Auto-generate Q/A pairs from document text
          2. Run each through the pipeline
          3. Score each with Ollama
          4. Aggregate metrics and save report

        Args:
            pipeline_fn: Callable that takes a query string and returns
                        dict with 'answer', 'sources' keys.
            document_text: Text to generate Q/A pairs from.
            num_samples: Number of Q/A pairs to evaluate.
            output_path: Where to save the JSON report.

        Returns:
            Aggregated evaluation report dict.
        """
        logger.info("Starting full evaluation run (%d samples)...", num_samples)
        start_time = time.time()

        # Generate test cases
        qa_pairs = self.generate_qa_pairs(document_text, count=num_samples)
        if not qa_pairs:
            return {"error": "Failed to generate Q/A pairs"}

        results: List[Dict[str, Any]] = []

        for i, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            expected = qa.get("expected_answer", "")

            logger.info("Evaluating sample %d/%d: '%s'", i + 1, len(qa_pairs), question[:60])

            # Run through the pipeline
            try:
                pipeline_result = pipeline_fn(question)
                answer = pipeline_result.get("answer", "")
                sources = pipeline_result.get("sources", [])
                context = "\n".join(s.get("text_preview", "") for s in sources)
            except Exception as e:
                logger.error("Pipeline failed for question '%s': %s", question[:60], e)
                answer = f"Pipeline error: {e}"
                context = ""

            # Evaluate with Ollama
            eval_result = self.evaluate_sample(
                question=question,
                context=context,
                answer=answer,
                expected_answer=expected,
            )
            eval_result["expected_answer"] = expected
            eval_result["actual_answer"] = answer[:500]
            results.append(eval_result)

        # Aggregate
        elapsed = round(time.time() - start_time, 2)
        metrics = ["faithfulness", "relevance", "completeness", "clarity", "weighted_average"]
        aggregated = {}
        for m in metrics:
            scores = [r.get(m, 0) for r in results if r.get(m, 0) > 0]
            aggregated[f"avg_{m}"] = round(sum(scores) / len(scores), 2) if scores else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "num_samples": len(results),
            "elapsed_seconds": elapsed,
            "aggregated_scores": aggregated,
            "detailed_results": results,
        }

        # Save report
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("Evaluation report saved to '%s'.", output_path)
        except Exception as e:
            logger.error("Failed to save evaluation report: %s", e)

        logger.info(
            "Evaluation complete in %.1fs — Avg scores: F=%.2f R=%.2f C=%.2f Cl=%.2f Overall=%.2f",
            elapsed, aggregated.get("avg_faithfulness", 0),
            aggregated.get("avg_relevance", 0),
            aggregated.get("avg_completeness", 0),
            aggregated.get("avg_clarity", 0),
            aggregated.get("avg_weighted_average", 0),
        )

        return report
