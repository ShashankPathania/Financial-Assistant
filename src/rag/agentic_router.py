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
    "route": "A", "B", or "C",
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

Route C (Conversational/Greeting): The query is a simple greeting, pleasantry, conversational remark, or a direct question about who you are.
Examples: "hello", "hi", "how are you?", "thanks that was helpful"

If a ticker is mentioned or implied, extract it. If dates are mentioned, extract the range.
If no dates are mentioned but market data is needed, use the last 6 months.

User Query: {query}"""

    AMBIGUITY_PROMPT = """You are a financial query analyzer.
Analyze the user's query and determine if it requires a specific company name or document context to be answered accurately. 
Generic financial questions (e.g., "What is EBITDA?", "How do SEC filings work?") DO NOT require a company name.
Specific performance data questions (e.g., "What was the Q3 profit?", "Who is the CEO?", "What are the core risks?") DO require a company name.

Respond ONLY with a JSON object:
{{
    "needs_company": true or false,
    "reasoning": "brief explanation"
}}

User Query: {query}"""

    GENERATION_PROMPT = """You are a senior financial analyst at a top-tier investment bank. 
Provide a comprehensive, professional analysis based on the context provided.

INSTRUCTIONS:
- Base your answer STRICTLY on the provided context
- If market data is available, integrate it with document findings
- Use specific numbers, dates, and facts from the context
- If the context contains OCR or vision references like "Text extracted from image", carefully read the pipe (|) delimited values as tabular data to find numerical answers
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

    def _call_ollama_fallback(self, prompt: str, is_json: bool = False, max_tokens: int = 2000) -> str:
        """Fallback to local Ollama instance if Groq fails."""
        import requests
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        try:
            logger.info("Calling local Ollama model: %s", ollama_model)
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1 if is_json else 0.3,
                    "num_predict": max_tokens
                }
            }
            if is_json:
                payload["format"] = "json"
                
            response = requests.post(ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error("Ollama fallback failed: %s", e)
            return ""

    # ------------------------------------------------------------------ #
    #  Query Classification
    # ------------------------------------------------------------------ #
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Use Groq to classify the query into Route A or Route B, with Ollama fallback."""
        prompt = self.ROUTE_CLASSIFICATION_PROMPT.format(query=query)
        text = ""

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    timeout=15,
                )
                text = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("Groq classification failed: %s", e)

        if not text:
            text = self._call_ollama_fallback(prompt, is_json=True, max_tokens=300)

        if not text:
            return {"route": "A", "reasoning": "All LLMs failed", "ticker": None, "date_start": None, "date_end": None}

        try:
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

    def _get_available_companies(self) -> List[str]:
        """Read parent_index.json to see what companies/documents we have."""
        try:
            parent_index_path = "data/chromadb_store/parent_index.json"
            if not os.path.exists(parent_index_path):
                return []
                
            with open(parent_index_path, "r") as f:
                index = json.load(f)
                
            # Extract unique source filenames
            sources = set()
            for doc in index.values():
                if "source" in doc:
                    sources.add(doc["source"])
            return list(sources)
        except Exception as e:
            logger.error("Failed to read available companies: %s", e)
            return []

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
        validation = self.security_manager.validate_user_input(query)
        if not validation.is_safe:
            return {
                "answer": validation.sanitized_input,  # error message
                "route": "BLOCKED",
                "sources": [],
                "market_data": {},
                "images": [],
            }
            
        sanitized_query = validation.sanitized_input

        # --- Step 2: Classify Query ---
        classification = self.classify_query(sanitized_query)
        route = classification.get("route", "A")
        ticker = classification.get("ticker")
        date_start = classification.get("date_start")
        date_end = classification.get("date_end")

        # --- Step 2b: Ambiguity / Clarification Guardrail ---
        # If it's not conversational and no ticker is provided, check if it needs one
        if route in ["A", "B"] and not ticker:
            ambiguity_prompt = self.AMBIGUITY_PROMPT.format(query=sanitized_query)
            text = ""
            
            if self.groq_client:
                try:
                    response = self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=[{"role": "user", "content": ambiguity_prompt}],
                        temperature=0.1,
                        max_tokens=100,
                    )
                    text = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning("Groq ambiguity check failed: %s", e)
                    
            if not text:
                text = self._call_ollama_fallback(ambiguity_prompt, is_json=True, max_tokens=100)
                
            if text:
                try:
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                    
                    ambiguity = json.loads(text.strip())
                    
                    # If it needs a company but doesn't have one, ask for clarification!
                    if ambiguity.get("needs_company"):
                        available_docs = self._get_available_companies()
                        
                        if available_docs:
                            doc_list = "\n".join([f"- {doc}" for doc in available_docs])
                            clarification = (
                                 "It looks like you are asking about specific financial data, but you didn't mention which company. "
                                 f"I currently have documents loaded for the following:\n\n{doc_list}\n\n"
                                 "Which one would you like me to analyze?"
                            )
                        else:
                            clarification = "It looks like you are asking about specific financial data, but I don't see a company name in your query, and my document database is currently empty!"
                        
                        return {
                            "answer": clarification,
                            "route": "CLARIFICATION",
                            "route_reasoning": ambiguity.get("reasoning", "Ambiguous query lacking company name."),
                            "sources": [],
                            "market_data": {},
                            "images": []
                        }
                except Exception as e:
                    logger.error("Ambiguity check parsing failed: %s", e)
                    # Fallback to proceed as normal if check fails
                    pass

        # --- Step 3: Handle Conversational Route ---
        if route == "C":
            return {
                "answer": "Hello! I am your Financial Analyst Copilot. I can help you analyze corporate documents like 10-Ks and look up historical market data. How can I help you today?",
                "route": "C",
                "route_reasoning": classification.get("reasoning", ""),
                "sources": [],
                "market_data": {},
                "images": [],
            }

        # --- Step 4: Retrieve Documents ---
        retrieval_results = self.retriever.retrieve(sanitized_query, top_k=5)
        
        # --- Step 4b: Empty Results Fallback ---
        if route == "A" and not retrieval_results:
            return {
                "answer": "I could not find any relevant information in the ingested documents to answer your question.",
                "route": "A",
                "route_reasoning": "No relevant context found above similarity threshold.",
                "sources": [],
                "market_data": {},
                "images": [],
            }

        # Sanitize retrieved contexts
        document_texts = [r.get("parent_text", r.get("child_text", "")) for r in retrieval_results]
        sanitized_contexts = self.security_manager.sanitize_context_chunks(document_texts)
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
        secure_prompt = self.security_manager.create_secure_prompt(
            user_query=sanitized_query,
            document_context=document_context or "No relevant documents found in the knowledge base.",
            market_context=market_context_str
        )
        
        answer = self._generate_response(
            secure_prompt=secure_prompt,
            chat_history=chat_history,
        )

        # Output Validation and Sanitization
        is_safe_response = self.security_manager.validate_response(answer)
        if not is_safe_response:
             logger.warning("Generation output blocked by security validator.")
             answer = "I cannot fulfill that request or output this response due to safety restrictions."
        else:
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
        secure_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate the final response using Groq, fallback to Ollama."""
        messages: List[Dict[str, str]] = []

        # Include chat history for conversational context
        if chat_history:
            for msg in chat_history[-6:]:  # last 3 turns
                messages.append(msg)

        messages.append({"role": "user", "content": secure_prompt})
        
        answer = ""

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    timeout=30,
                )
                answer = response.choices[0].message.content.strip()
                logger.info("Response generated via Groq (%d chars).", len(answer))
            except Exception as e:
                logger.error("Groq generation failed: %s", e)
                
        if not answer:
            # Reconstruct prompt for completion since Ollama generate API takes string
            ollama_prompt = ""
            for msg in messages:
                ollama_prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"
            
            answer = self._call_ollama_fallback(ollama_prompt.strip(), is_json=False, max_tokens=2000)
            if answer:
                logger.info("Response generated via Ollama (%d chars).", len(answer))

        if not answer:
            return "⚠️ Generation error: Both Groq and Local Fallback failed to return a response."
            
        return answer
