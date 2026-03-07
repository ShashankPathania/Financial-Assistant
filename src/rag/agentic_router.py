"""
Agentic Router
================
Uses Groq LLM to classify user queries into:
  - Route A (Document Only): triggers Advanced Retrieval
  - Route B (Market Context): extracts Ticker + Date Range, calls yfinance,
    AND triggers Advanced Retrieval, then merges both into the final prompt.

Also handles the final generation step with the combined context.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

from src.rag.advanced_retriever import AdvancedRetriever
from src.rag.security_manager import EnterpriseSecurityManager

load_dotenv()
logger = logging.getLogger(__name__)


class AgenticRouter:
    """
    Agentic query router and generation pipeline.

    1. Classify query → Route A or Route B
    2. Execute retrieval (and optionally yfinance)
    3. Merge context and generate final response via Groq
    """

    ROUTE_CLASSIFICATION_PROMPT = """You are a financial query router. Analyze the user's query and classify it.

Respond with a JSON object (and ONLY JSON, no markdown):
{{
    "route": "A" or "B",
    "reasoning": "brief explanation",
    "ticker": "stock ticker if mentioned, else null",
    "date_start": "YYYY-MM-DD or null",
    "date_end": "YYYY-MM-DD or null"
}}

Route A (Document Only): The query is about information that can be found in corporate financial documents, 
annual reports, regulatory filings, risk factors, financial statements, etc. 
Examples: "What are the risk factors?", "What was the revenue breakdown?", "Summarize the MD&A section."

Route B (Market Context): The query references stock market performance, stock price reactions, 
trading volume, market trends, or requires combining document info with market data.
Examples: "How did the stock react to Q3 results?", "What's the stock performance since the annual report?",
"Compare the revenue growth with stock returns."

If a ticker is mentioned or implied, extract it. If dates are mentioned, extract the range.
If no dates are mentioned but market data is needed, use the last 6 months.

User Query: {query}"""

    GENERATION_PROMPT = """You are a senior financial analyst at a top-tier investment bank. 
Provide a comprehensive, professional analysis based on the context provided.

INSTRUCTIONS:
- Base your answer STRICTLY on the provided context
- If market data is available, integrate it with document findings
- Use specific numbers, dates, and facts from the context
- If the context doesn't contain enough information, say so clearly
- Structure your response with clear sections when appropriate
- Be concise but thorough

{market_context}

DOCUMENT CONTEXT:
{document_context}

USER QUESTION: {query}

PROFESSIONAL ANALYSIS:"""

    def __init__(
        self,
        retriever: AdvancedRetriever,
        security_manager: Optional[EnterpriseSecurityManager] = None,
        groq_model: str = "",
    ):
        self.retriever = retriever
        self.security_manager = security_manager or EnterpriseSecurityManager()

        api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        self.groq_model = groq_model or os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

        logger.info("AgenticRouter initialized (model=%s)", self.groq_model)

    # ------------------------------------------------------------------ #
    #  Query Classification
    # ------------------------------------------------------------------ #
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Use Groq to classify the query into Route A or Route B."""
        if not self.groq_client:
            logger.warning("Groq client unavailable — defaulting to Route A.")
            return {"route": "A", "reasoning": "Groq unavailable", "ticker": None,
                    "date_start": None, "date_end": None}

        try:
            prompt = self.ROUTE_CLASSIFICATION_PROMPT.format(query=query)
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                timeout=15,
            )

            text = response.choices[0].message.content.strip()
            # Clean up potential markdown wrapping
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            classification = json.loads(text)
            logger.info("Query classified as Route %s: %s", classification.get("route"),
                        classification.get("reasoning"))
            return classification

        except json.JSONDecodeError as e:
            logger.error("Failed to parse classification JSON: %s", e)
            return {"route": "A", "reasoning": "JSON parse error", "ticker": None,
                    "date_start": None, "date_end": None}
        except Exception as e:
            logger.error("Query classification failed: %s", e)
            return {"route": "A", "reasoning": str(e), "ticker": None,
                    "date_start": None, "date_end": None}

    # ------------------------------------------------------------------ #
    #  yfinance Market Data
    # ------------------------------------------------------------------ #
    def fetch_market_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch historical stock data from yfinance."""
        if not ticker:
            return {}

        try:
            # Default date range: last 6 months
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

            logger.info("Fetching yfinance data: %s from %s to %s", ticker, start_date, end_date)

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                logger.warning("No market data found for %s.", ticker)
                return {"ticker": ticker, "error": "No data found"}

            # Compute summary statistics
            info = {}
            try:
                info = stock.info
            except Exception:
                pass

            market_data = {
                "ticker": ticker,
                "period": f"{start_date} to {end_date}",
                "start_price": round(float(hist["Close"].iloc[0]), 2),
                "end_price": round(float(hist["Close"].iloc[-1]), 2),
                "high": round(float(hist["High"].max()), 2),
                "low": round(float(hist["Low"].min()), 2),
                "avg_volume": int(hist["Volume"].mean()),
                "total_return_pct": round(
                    ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100, 2
                ),
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "data_points": len(hist),
            }

            logger.info("Market data fetched: %s return=%.2f%%", ticker, market_data["total_return_pct"])
            return market_data

        except Exception as e:
            logger.error("yfinance fetch failed for %s: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def _format_market_context(self, market_data: Dict[str, Any]) -> str:
        """Format market data into a readable context string."""
        if not market_data or market_data.get("error"):
            return ""

        return (
            f"MARKET DATA for {market_data.get('company_name', market_data['ticker'])} "
            f"({market_data['ticker']}):\n"
            f"  Period: {market_data['period']}\n"
            f"  Start Price: ${market_data['start_price']}\n"
            f"  End Price: ${market_data['end_price']}\n"
            f"  Total Return: {market_data['total_return_pct']}%\n"
            f"  52-Week High (period): ${market_data['high']}\n"
            f"  52-Week Low (period): ${market_data['low']}\n"
            f"  Average Daily Volume: {market_data['avg_volume']:,}\n"
            f"  Sector: {market_data.get('sector', 'N/A')}\n"
            f"  Market Cap: {market_data.get('market_cap', 'N/A')}\n"
        )

    # ------------------------------------------------------------------ #
    #  Full Pipeline: Route → Retrieve → Generate
    # ------------------------------------------------------------------ #
    def process_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Full agentic pipeline:
          1. Security validation
          2. Classify query
          3. Retrieve documents (and optionally market data)
          4. Generate response

        Returns:
            dict with 'answer', 'route', 'sources', 'market_data', 'images'
        """
        # --- Step 1: Security Validation ---
        is_safe, sanitized_query = self.security_manager.validate_input(query)
        if not is_safe:
            return {
                "answer": sanitized_query,  # error message
                "route": "BLOCKED",
                "sources": [],
                "market_data": {},
                "images": [],
            }

        # --- Step 2: Classify Query ---
        classification = self.classify_query(sanitized_query)
        route = classification.get("route", "A")
        ticker = classification.get("ticker")
        date_start = classification.get("date_start")
        date_end = classification.get("date_end")

        # --- Step 3: Retrieve Documents ---
        retrieval_results = self.retriever.retrieve(sanitized_query, top_k=5)

        # Sanitize retrieved contexts
        document_texts = [r.get("parent_text", r.get("child_text", "")) for r in retrieval_results]
        sanitized_contexts = self.security_manager.sanitize_context(document_texts)
        document_context = "\n\n---\n\n".join(
            f"[Source: {r.get('source', 'unknown')}, Page {r.get('page', '?')}]\n{ctx}"
            for r, ctx in zip(retrieval_results, sanitized_contexts)
        )

        # Collect image paths from results
        images = [
            r.get("image_path", "")
            for r in retrieval_results
            if r.get("image_path")
        ]

        # --- Step 3b: Fetch Market Data (Route B only) ---
        market_data = {}
        market_context_str = ""
        if route == "B" and ticker:
            market_data = self.fetch_market_data(ticker, date_start, date_end)
            market_context_str = self._format_market_context(market_data)
            if market_context_str:
                market_context_str = f"MARKET DATA CONTEXT:\n{market_context_str}"

        # --- Step 4: Generate Response ---
        answer = self._generate_response(
            query=sanitized_query,
            document_context=document_context or "No relevant documents found in the knowledge base.",
            market_context=market_context_str,
            chat_history=chat_history,
        )

        # Output sanitization
        answer = self.security_manager.sanitize_output(answer)

        # Build sources list
        sources = [
            {
                "source": r.get("source", "unknown"),
                "page": r.get("page", 0),
                "text_preview": r.get("child_text", "")[:200],
                "score": r.get("score", 0),
                "image_path": r.get("image_path", ""),
            }
            for r in retrieval_results
        ]

        return {
            "answer": answer,
            "route": route,
            "route_reasoning": classification.get("reasoning", ""),
            "sources": sources,
            "market_data": market_data,
            "images": images,
        }

    # ------------------------------------------------------------------ #
    #  LLM Generation
    # ------------------------------------------------------------------ #
    def _generate_response(
        self,
        query: str,
        document_context: str,
        market_context: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate the final response using Groq."""
        if not self.groq_client:
            return (
                "⚠️ Groq API key is not configured. Please set GROQ_API_KEY in your .env file.\n\n"
                f"Retrieved {len(document_context)} chars of document context for your query."
            )

        prompt = self.GENERATION_PROMPT.format(
            query=query,
            document_context=document_context,
            market_context=market_context,
        )

        messages: List[Dict[str, str]] = []

        # Include chat history for conversational context
        if chat_history:
            for msg in chat_history[-6:]:  # last 3 turns
                messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                timeout=30,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Response generated (%d chars).", len(answer))
            return answer

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return f"⚠️ Generation error: {str(e)}. Please try again."
