"""
Multi-Modal Corporate Finance & Market Analyst Copilot
========================================================
Streamlit Frontend Application

Pages:
  1. About — Architecture overview & system description
  2. Ingest Documents — Scrape PDFs or upload manually, run ingestion pipeline
  3. Chatbot Analysis — Interactive chat with agentic routing & source display
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Fix for Streamlit Cloud SQLite version issue with ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
#  Setup
# ------------------------------------------------------------------ #
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ------------------------------------------------------------------ #
#  Page Configuration
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Finance Analyst Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #0d0d2b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0ff !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1rem;
        font-weight: 500;
    }

    /* ---- Hero Header ---- */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .hero-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero-header p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    /* ---- Stat Cards ---- */
    .stat-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        color: #e0e0ff;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }

    /* ---- Feature Cards ---- */
    .feature-card {
        background: rgba(30, 30, 63, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .feature-card h3 {
        color: #667eea;
        margin-top: 0;
    }
    .feature-card p {
        color: #c0c0e0;
        line-height: 1.6;
    }

    /* ---- Chat Styling ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* ---- Route Badge ---- */
    .route-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .route-a { background: #1b4332; color: #95d5b2; }
    .route-b { background: #3c1642; color: #c77dff; }

    /* ---- Progress ---- */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* ---- Source Expander ---- */
    .source-chip {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        color: #c0c0e0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Lazy-Loaded Pipeline Components (cached)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Loading embedding model & vector store...")
def get_chunking_engine():
    from src.ingestion.chunking_engine import ChunkingEngine
    return ChunkingEngine(
        chromadb_path=str(PROJECT_ROOT / "data" / "chromadb_store"),
        parent_index_path=str(PROJECT_ROOT / "data" / "chromadb_store" / "parent_index.json"),
    )


@st.cache_resource(show_spinner="Initializing retriever...")
def get_retriever():
    chunking_engine = get_chunking_engine()
    from src.rag.advanced_retriever import AdvancedRetriever
    return AdvancedRetriever(chunking_engine=chunking_engine)


@st.cache_resource(show_spinner="Initializing agentic router...")
def get_router():
    retriever = get_retriever()
    from src.rag.agentic_router import AgenticRouter
    from src.rag.security_manager import EnterpriseSecurityManager
    return AgenticRouter(
        retriever=retriever,
        security_manager=EnterpriseSecurityManager(),
    )


@st.cache_resource(show_spinner="Loading PDF parser...")
def get_parser():
    from src.ingestion.multi_modal_parser import MultiModalParser
    return MultiModalParser(images_dir=str(PROJECT_ROOT / "data" / "extracted_images"))


# ------------------------------------------------------------------ #
#  Sidebar Navigation
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 📊 Finance Copilot")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 About", "📥 Ingest Documents", "💬 Chatbot Analysis", "📊 Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Status indicators
    st.markdown("### System Status")
    api_key_set = bool(os.getenv("GROQ_API_KEY", "")) and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here"
    st.markdown(f"{'✅' if api_key_set else '❌'} Groq API Key")

    try:
        ce = get_chunking_engine()
        doc_count = ce.get_collection_count()
        st.markdown(f"✅ ChromaDB ({doc_count} chunks)")
    except Exception:
        doc_count = 0
        st.markdown("⚠️ ChromaDB not initialized")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.8rem;'>"
        "Built with Groq • ChromaDB • Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ================================================================== #
#  PAGE 1: ABOUT
# ================================================================== #
if page == "🏠 About":
    st.markdown("""
    <div class="hero-header">
        <h1>📊 Multi-Modal Corporate Finance &amp; Market Analyst Copilot</h1>
        <p>An enterprise-grade RAG system for financial document analysis with agentic market data routing</p>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">6</div>
            <div class="stat-label">Pipeline Stages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">3</div>
            <div class="stat-label">LLM Models</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{doc_count}</div>
            <div class="stat-label">Indexed Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">2</div>
            <div class="stat-label">Query Routes</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture description
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Multi-Modal PDF Ingestion</h3>
            <p>
            Layout-aware parsing extracts text, tables, and charts from financial PDFs.
            The <code>unstructured</code> library handles complex layouts while Groq Vision
            (<code>llama-3.2-90b-vision-preview</code>) generates factual summaries of
            extracted visuals — turning charts into searchable text.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Agentic Query Router</h3>
            <p>
            Every query is classified by Groq (<code>llama-3.3-70b-versatile</code>) into:<br>
            <strong>Route A</strong> (Document Only) — pure RAG retrieval<br>
            <strong>Route B</strong> (Market Context) — RAG + live stock data from <code>yfinance</code>,
            giving you both fundamental analysis and market performance in one answer.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Enterprise Security</h3>
            <p>
            Three-layer guardrails: regex-based jailbreak detection on inputs,
            HTML/XML tag stripping on retrieved contexts, and PII redaction on outputs.
            All queries pass through the <code>EnterpriseSecurityManager</code> before
            reaching any LLM.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="feature-card">
            <h3>📐 Parent-Child Chunking</h3>
            <p>
            Documents are split into parent chunks (~2000 chars, broad narrative) and
            child chunks (~500 chars, specific details). Child chunks are embedded with
            <code>all-MiniLM-L6-v2</code> and stored in <strong>ChromaDB</strong>.
            Retrieval matches children, then fetches parent context for comprehensive answers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>⚡ HyDE + Multi-Query Retrieval</h3>
            <p>
            For every query, the system generates a hypothetical answer (HyDE) and 3
            alternative phrasings (Multi-Query Expansion), embedding and searching
            all variations. This dramatically improves recall for complex financial queries.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>📊 Hybrid LLM Strategy</h3>
            <p>
            <strong>Groq</strong> handles all user-facing tasks (routing, generation, HyDE).
            <strong>Groq Vision</strong> summarizes extracted charts during ingestion.
            <strong>Ollama</strong> (local llama3.1) runs the offline LLM-as-a-Judge
            evaluator — saving API limits for heavy evaluation workloads.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Data flow diagram
    st.markdown("### 🔄 System Architecture Flow")
    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                        USER QUERY                                     │
    │                           │                                           │
    │                    ┌──────▼──────┐                                    │
    │                    │  Security   │ ◄── Jailbreak Detection            │
    │                    │  Manager    │ ◄── Input Validation               │
    │                    └──────┬──────┘                                    │
    │                           │                                           │
    │                    ┌──────▼──────┐                                    │
    │                    │   Agentic   │ ◄── Groq LLM Classification       │
    │                    │   Router    │                                    │
    │                    └──┬─────┬────┘                                    │
    │             Route A   │     │  Route B                               │
    │           ┌───────────┘     └───────────┐                            │
    │           │                             │                            │
    │    ┌──────▼──────┐              ┌───────▼──────┐                     │
    │    │  Advanced   │              │   yfinance   │                     │
    │    │  Retriever  │              │  Market Data │                     │
    │    │  (HyDE +    │              └───────┬──────┘                     │
    │    │  MultiQuery)│                      │                            │
    │    └──────┬──────┘              ┌───────▼──────┐                     │
    │           │                     │  Advanced    │                     │
    │           │                     │  Retriever   │                     │
    │           │                     └───────┬──────┘                     │
    │           └──────────┬─────────────────┘                             │
    │                      │                                               │
    │               ┌──────▼──────┐                                        │
    │               │   Groq LLM  │ ◄── Final Generation                  │
    │               │  Generation │                                        │
    │               └──────┬──────┘                                        │
    │                      │                                               │
    │               ┌──────▼──────┐                                        │
    │               │   Output    │ ◄── PII Redaction                     │
    │               │   Security  │                                        │
    │               └──────┬──────┘                                        │
    │                      ▼                                               │
    │                 RESPONSE + SOURCES                                    │
    └────────────────────────────────────────────────────────────────────────┘
    ```
    """)


# ================================================================== #
#  PAGE 2: INGEST DOCUMENTS
# ================================================================== #
elif page == "📥 Ingest Documents":
    st.markdown("""
    <div class="hero-header">
        <h1>📥 Document Ingestion</h1>
        <p>Scrape financial PDFs from the web or upload documents manually for analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌐 Web Scraper", "📄 Manual Upload"])

    # ---- Tab 1: Web Scraper ----
    with tab1:
        st.markdown("#### Scrape Financial Documents")
        st.info(
            "Download filings from international financial portals or any direct URL."
        )

        # Portal selection
        portal_options = {
            "🇺🇸 SEC EDGAR (US)": "SEC_EDGAR",
            "🇬🇧 Companies House (UK)": "COMPANIES_HOUSE",
            "🇮🇳 BSE India": "BSE_INDIA",
            "🔗 Direct URL Download": "DIRECT_URL",
        }
        selected_portal_label = st.selectbox("Select Portal", list(portal_options.keys()))
        selected_portal = portal_options[selected_portal_label]

        # Dynamic input fields based on portal
        scrape_ticker = ""
        scrape_year = ""
        scrape_type = "10-K"
        custom_url = ""
        company_number = ""

        if selected_portal == "SEC_EDGAR":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                scrape_ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="AAPL")
            with col_b:
                scrape_year = st.text_input("Filing Year", value="2024", placeholder="2024")
            with col_c:
                scrape_type = st.selectbox("Report Type", ["10-K", "10-Q", "8-K"])

        elif selected_portal == "COMPANIES_HOUSE":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                scrape_ticker = st.text_input("Company Name", value="", placeholder="Vodafone Group")
            with col_b:
                company_number = st.text_input("Company Number (optional)", value="", placeholder="01833679")
            with col_c:
                scrape_year = st.text_input("Filing Year", value="2024", placeholder="2024")
            st.caption("💡 Free API key from [Companies House Developer](https://developer.company-information.service.gov.uk). Set `COMPANIES_HOUSE_API_KEY` in `.env`.")

        elif selected_portal == "BSE_INDIA":
            col_a, col_b = st.columns(2)
            with col_a:
                scrape_ticker = st.text_input("Company Name / Scrip Code", value="", placeholder="Reliance Industries")
            with col_b:
                scrape_year = st.text_input("Year", value="2024", placeholder="2024")

        elif selected_portal == "DIRECT_URL":
            custom_url = st.text_input(
                "📎 Paste any URL to a PDF or HTML filing",
                value="",
                placeholder="https://example.com/annual-report-2024.pdf",
            )
            scrape_ticker = st.text_input("Label (for filename)", value="company", placeholder="company_name")

        if st.button("🚀 Start Scraping", use_container_width=True):
            with st.spinner(f"Downloading from {selected_portal_label}..."):
                try:
                    from src.scraper.financial_scraper import scrape_financial_pdfs
                    progress = st.progress(0, text=f"Connecting to {selected_portal_label}...")

                    files = scrape_financial_pdfs(
                        ticker=scrape_ticker,
                        year=scrape_year,
                        report_type=scrape_type,
                        portal=selected_portal,
                        custom_url=custom_url,
                        company_number=company_number,
                    )
                    progress.progress(50, text=f"Downloaded {len(files)} file(s). Starting ingestion...")

                    if files:
                        parser = get_parser()
                        engine = get_chunking_engine()

                        for i, filepath in enumerate(files):
                            progress.progress(
                                50 + int(50 * (i + 1) / len(files)),
                                text=f"Parsing {Path(filepath).name}...",
                            )
                            elements = parser.parse_pdf(filepath)
                            if elements:
                                engine.process_elements(elements, source_file=Path(filepath).name)

                        progress.progress(100, text="✅ Ingestion complete!")
                        st.success(f"Successfully ingested {len(files)} document(s)!")
                        st.balloons()
                    else:
                        progress.progress(100, text="⚠️ No PDFs found.")
                        st.warning("No PDF files were discovered. Try a different ticker or custom URL.")

                except Exception as e:
                    st.error(f"Scraping error: {e}")
                    logger.error("Scraping error: %s", e, exc_info=True)

    # ---- Tab 2: Manual Upload ----
    with tab2:
        st.markdown("#### Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF financial documents for ingestion.",
        )

        if uploaded_files and st.button("📂 Process Uploaded Files", use_container_width=True):
            parser = get_parser()
            engine = get_chunking_engine()

            progress = st.progress(0, text="Starting ingestion...")
            total = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    int((idx / total) * 100),
                    text=f"Processing {uploaded_file.name}...",
                )

                # Save to raw_pdfs directory
                save_path = PROJECT_ROOT / "data" / "raw_pdfs" / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Parse and chunk
                elements = parser.parse_pdf(str(save_path))
                if elements:
                    result = engine.process_elements(elements, source_file=uploaded_file.name)
                    st.info(
                        f"📄 **{uploaded_file.name}**: "
                        f"{result['parents_added']} parents, {result['children_added']} children"
                    )
                else:
                    st.warning(f"⚠️ No extractable content found in {uploaded_file.name}")

            progress.progress(100, text="✅ All files processed!")
            st.success(f"Successfully ingested {total} document(s)!")
            st.balloons()

    # ---- Collection Info ----
    st.markdown("---")
    st.markdown("#### 📊 Current Index Status")
    try:
        ce = get_chunking_engine()
        count = ce.get_collection_count()
        parent_count = len(ce.parent_index)

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Child Chunks in ChromaDB", count)
        col_s2.metric("Parent Chunks in Index", parent_count)
        col_s3.metric("Unique Sources", len(set(
            p.get("source", "") for p in ce.parent_index.values()
        )))
    except Exception as e:
        st.warning(f"Could not load index status: {e}")


# ================================================================== #
#  PAGE 3: CHATBOT ANALYSIS
# ================================================================== #
elif page == "💬 Chatbot Analysis":
    st.markdown("""
    <div class="hero-header">
        <h1>💬 Financial Analyst Chat</h1>
        <p>Ask questions about your ingested documents — with automatic market data integration</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources in expander if available
            if msg.get("sources"):
                with st.expander("📚 View Sources & Context"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"<div class='source-chip'>"
                            f"📄 <strong>{src.get('source', 'unknown')}</strong> "
                            f"— Page {src.get('page', '?')} "
                            f"(Score: {src.get('score', 0):.3f})</div>",
                            unsafe_allow_html=True,
                        )
                        if src.get("text_preview"):
                            st.caption(src["text_preview"])

                        # Render extracted images
                        if src.get("image_path") and os.path.exists(src["image_path"]):
                            st.image(src["image_path"], caption="Extracted visual", width=400)

            # Show market data if available
            if msg.get("market_data") and not msg["market_data"].get("error"):
                with st.expander("📈 Market Data"):
                    md = msg["market_data"]
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Start Price", f"${md.get('start_price', 'N/A')}")
                    mc2.metric("End Price", f"${md.get('end_price', 'N/A')}")
                    mc3.metric("Return", f"{md.get('total_return_pct', 0)}%")
                    mc4.metric("Avg Volume", f"{md.get('avg_volume', 0):,}")

    # Chat input
    if prompt := st.chat_input("Ask about your financial documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing..."):
                try:
                    router = get_router()
                    result = router.process_query(
                        query=prompt,
                        chat_history=st.session_state.chat_history,
                    )

                    answer = result.get("answer", "No response generated.")
                    route = result.get("route", "A")
                    sources = result.get("sources", [])
                    market_data = result.get("market_data", {})
                    images = result.get("images", [])

                    # Route badge
                    if route == "A":
                        st.markdown(
                            '<span class="route-badge route-a">📄 Route A — Document Analysis</span>',
                            unsafe_allow_html=True,
                        )
                    elif route == "B":
                        st.markdown(
                            '<span class="route-badge route-b">📈 Route B — Market + Document</span>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(answer)

                    # Sources expander
                    if sources:
                        with st.expander("📚 View Sources & Context"):
                            for src in sources:
                                st.markdown(
                                    f"<div class='source-chip'>"
                                    f"📄 <strong>{src.get('source', 'unknown')}</strong> "
                                    f"— Page {src.get('page', '?')} "
                                    f"(Score: {src.get('score', 0):.3f})</div>",
                                    unsafe_allow_html=True,
                                )
                                if src.get("text_preview"):
                                    st.caption(src["text_preview"])
                                if src.get("image_path") and os.path.exists(src["image_path"]):
                                    st.image(src["image_path"], caption="Extracted visual", width=400)

                    # Market data expander
                    if market_data and not market_data.get("error"):
                        with st.expander("📈 Market Data"):
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Start Price", f"${market_data.get('start_price', 'N/A')}")
                            mc2.metric("End Price", f"${market_data.get('end_price', 'N/A')}")
                            mc3.metric("Return", f"{market_data.get('total_return_pct', 0)}%")
                            mc4.metric("Avg Volume", f"{market_data.get('avg_volume', 0):,}")

                    # Save message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "market_data": market_data,
                    })

                    # Update chat history for conversational memory
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                    # Keep only the last 10 turns in history
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]

                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error("Chat error: %s", e, exc_info=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

    # Sidebar controls for chat
    with st.sidebar:
        st.markdown("### Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()


# ================================================================== #
#  PAGE 4: EVALUATION
# ================================================================== #
elif page == "📊 Evaluation":
    st.markdown("""
    <div class="hero-header">
        <h1>📊 RAG Pipeline Evaluation</h1>
        <p>Measure retrieval quality with Precision, Recall, MRR and LLM-judged metrics</p>
    </div>
    """, unsafe_allow_html=True)

    # Check prerequisites
    try:
        ce = get_chunking_engine()
        chunk_count = ce.get_collection_count()
        parent_count = len(ce.parent_index)
    except Exception:
        chunk_count = 0
        parent_count = 0

    if chunk_count == 0:
        st.warning("⚠️ No documents ingested yet. Go to **📥 Ingest Documents** first.")
    else:
        st.markdown(f"**📚 Index contains {chunk_count} chunks from {parent_count} parent blocks.**")

        # ---- Evaluation Controls ----
        st.markdown("---")
        st.markdown("### Configure Evaluation")

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            num_queries = st.slider("Number of test queries", 2, 10, 5)
        with col_e2:
            k_value = st.slider("K (for P@K and R@K)", 3, 20, 5)
        with col_e3:
            top_n_retrieve = st.slider("Chunks to retrieve per query", 5, 20, 10)

        if st.button("🚀 Run Evaluation", use_container_width=True):
            try:
                from src.evaluation.local_llm_evaluator import LocalLLMEvaluator

                # Check if Ollama is reachable before instantiating the evaluator
                import requests
                try:
                    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    requests.get(f"{ollama_base_url}/api/tags", timeout=2)
                except requests.RequestException:
                    st.error("⚠️ Ollama is not reachable. The local evaluator requires a running Ollama instance (which cannot run on Streamlit Community Cloud).")
                    st.stop()

                evaluator = LocalLLMEvaluator()
                retriever = get_retriever()
                router = get_router()
                engine = get_chunking_engine()

                # ---- Step 1: Generate test queries from ingested content ----
                st.markdown("---")
                status = st.status("Running evaluation pipeline...", expanded=True)

                with status:
                    st.write("🧠 Generating test queries from documents...")

                    # Sample parents evenly across ALL sources (1 random chunk per source)
                    import random
                    from collections import defaultdict
                    sources_map = defaultdict(list)
                    for p in engine.parent_index.values():
                        sources_map[p.get("source", "unknown")].append(p)

                    sample_parents = []
                    for source, parents in sources_map.items():
                        if parents:
                            sample_parents.append(random.choice(parents))

                    # Shuffle to distribute representation
                    random.shuffle(sample_parents)

                    st.write(f"📚 Sampling from {len(sources_map)} source(s): {', '.join(sources_map.keys())}")
                    
                    # Take up to 4 sources, 1000 chars each to fit Ollama's 4000 char prompt limit
                    sample_text = "\n\n".join(p.get("text", "")[:1000] for p in sample_parents[:4])

                    qa_pairs = evaluator.generate_qa_pairs(sample_text, count=num_queries)

                    if not qa_pairs:
                        st.error("❌ Failed to generate test queries. Make sure Ollama is running with llama3.1:latest.")
                        st.stop()

                    st.write(f"✅ Generated {len(qa_pairs)} test queries.")

                    # ---- Step 2: Run retrieval + metrics for each query ----
                    st.write("🔍 Running retrieval and computing metrics...")

                    all_retrieval_metrics = []
                    all_generation_metrics = []
                    per_query_details = []

                    progress = st.progress(0)

                    for i, qa in enumerate(qa_pairs):
                        question = qa.get("question", "")
                        expected = qa.get("expected_answer", "")
                        progress.progress((i + 1) / len(qa_pairs), text=f"Query {i+1}/{len(qa_pairs)}: {question[:60]}...")

                        # --- Retrieve chunks ---
                        try:
                            retrieval_result = retriever.retrieve(question, top_k=top_n_retrieve)
                            retrieved_chunks = []
                            for j, item in enumerate(retrieval_result):
                                retrieved_chunks.append({
                                    "id": item.get("parent_id", str(j)),
                                    "text": item.get("child_text", "") or item.get("parent_text", ""),
                                })
                        except Exception as ret_err:
                            logger.error("Retrieval failed: %s", ret_err)
                            retrieved_chunks = []

                        # --- Compute retrieval metrics (P@K, R@K, MRR) ---
                        if retrieved_chunks:
                            ret_metrics = evaluator.compute_retrieval_metrics(
                                query=question,
                                retrieved_chunks=retrieved_chunks,
                                k=k_value,
                            )
                        else:
                            ret_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

                        all_retrieval_metrics.append(ret_metrics)

                        # --- Run pipeline and evaluate generation quality ---
                        try:
                            pipeline_result = router.process_query(question)
                            answer = pipeline_result.get("answer", "")
                            sources = pipeline_result.get("sources", [])
                            context = "\n".join(s.get("text_preview", "") for s in sources)
                        except Exception:
                            answer = "Pipeline error"
                            context = ""

                        gen_metrics = evaluator.evaluate_sample(
                            question=question, context=context,
                            answer=answer, expected_answer=expected,
                        )
                        all_generation_metrics.append(gen_metrics)

                        per_query_details.append({
                            "Query": question[:80],
                            "P@K": ret_metrics["precision"],
                            "R@K": ret_metrics["recall"],
                            "MRR": ret_metrics["mrr"],
                            "Faithfulness": gen_metrics.get("faithfulness", 0),
                            "Relevance": gen_metrics.get("relevance", 0),
                            "Completeness": gen_metrics.get("completeness", 0),
                            "Clarity": gen_metrics.get("clarity", 0),
                            "Weighted Avg": gen_metrics.get("weighted_average", 0),
                        })

                    progress.progress(1.0, text="✅ Evaluation complete!")
                    st.write("✅ All queries evaluated.")

                # ---- Step 3: Compute aggregates ----
                avg_precision = sum(m["precision"] for m in all_retrieval_metrics) / len(all_retrieval_metrics)
                avg_recall = sum(m["recall"] for m in all_retrieval_metrics) / len(all_retrieval_metrics)
                avg_mrr = sum(m["mrr"] for m in all_retrieval_metrics) / len(all_retrieval_metrics)

                avg_faithfulness = sum(m.get("faithfulness", 0) for m in all_generation_metrics) / len(all_generation_metrics)
                avg_relevance = sum(m.get("relevance", 0) for m in all_generation_metrics) / len(all_generation_metrics)
                avg_completeness = sum(m.get("completeness", 0) for m in all_generation_metrics) / len(all_generation_metrics)
                avg_clarity = sum(m.get("clarity", 0) for m in all_generation_metrics) / len(all_generation_metrics)

                # ---- Step 4: Display Score Cards ----
                st.markdown("---")
                st.markdown("### 📊 Retrieval Quality Metrics")

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_precision:.2%}</div>
                        <div class="stat-label">Precision@{k_value}</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_recall:.2%}</div>
                        <div class="stat-label">Recall@{k_value}</div>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_mrr:.4f}</div>
                        <div class="stat-label">MRR (Mean Reciprocal Rank)</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🧠 LLM Generation Quality (1-5 scale)")

                gc1, gc2, gc3, gc4 = st.columns(4)
                with gc1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_faithfulness:.1f}</div>
                        <div class="stat-label">Faithfulness</div>
                    </div>""", unsafe_allow_html=True)
                with gc2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_relevance:.1f}</div>
                        <div class="stat-label">Relevance</div>
                    </div>""", unsafe_allow_html=True)
                with gc3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_completeness:.1f}</div>
                        <div class="stat-label">Completeness</div>
                    </div>""", unsafe_allow_html=True)
                with gc4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{avg_clarity:.1f}</div>
                        <div class="stat-label">Clarity</div>
                    </div>""", unsafe_allow_html=True)

                # ---- Step 5: Per-Query Breakdown Table ----
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Per-Query Breakdown")

                import pandas as pd
                df = pd.DataFrame(per_query_details)
                st.dataframe(
                    df.style.format({
                        "P@K": "{:.2%}",
                        "R@K": "{:.2%}",
                        "MRR": "{:.4f}",
                        "Faithfulness": "{:.1f}",
                        "Relevance": "{:.1f}",
                        "Completeness": "{:.1f}",
                        "Clarity": "{:.1f}",
                        "Weighted Avg": "{:.2f}",
                    }).background_gradient(
                        subset=["P@K", "R@K", "MRR"],
                        cmap="Greens", vmin=0, vmax=1,
                    ).background_gradient(
                        subset=["Faithfulness", "Relevance", "Completeness", "Clarity"],
                        cmap="Blues", vmin=1, vmax=5,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # ---- Save results to session state ----
                st.session_state["eval_results"] = {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "mrr": avg_mrr,
                    "faithfulness": avg_faithfulness,
                    "relevance": avg_relevance,
                    "completeness": avg_completeness,
                    "clarity": avg_clarity,
                    "details": per_query_details,
                }

                st.success(
                    f"✅ Evaluation complete — "
                    f"P@{k_value}: {avg_precision:.2%} | "
                    f"R@{k_value}: {avg_recall:.2%} | "
                    f"MRR: {avg_mrr:.4f}"
                )

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                logger.error("Evaluation error: %s", e, exc_info=True)
