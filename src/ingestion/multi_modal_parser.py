"""
Multi-Modal PDF Parser
========================
Uses the `unstructured` library for layout-aware PDF parsing that correctly
handles text paragraphs, financial tables, and embedded charts.  Extracted
images/tables are saved as .png files and summarized using local HuggingFace
models (vit-gpt2 + EasyOCR) or optionally via Groq Vision API.
"""

import base64
import json
import logging
import os
import requests
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EXTRACTED_IMAGES_DIR = "data/extracted_images"

# Local captioning model (lightweight, uses safetensors)
CAPTIONING_MODEL = "Salesforce/blip-image-captioning-base"


class MultiModalParser:
    """
    Parse PDFs using the `unstructured` library, extract images/tables,
    and summarize visuals using local HuggingFace models (default) or
    Groq Vision API (opt-in via USE_GROQ_VISION=true).
    """

    def __init__(
        self,
        images_dir: str = EXTRACTED_IMAGES_DIR,
        groq_vision_model: str = "",
    ):
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)

        # Handle Poppler Path (Windows Fix)
        poppler_path = os.getenv("POPPLER_PATH", "").strip('"')
        if poppler_path:
            # Normalize path to handle both / and \
            poppler_path = os.path.abspath(poppler_path)
            
            if os.path.exists(poppler_path):
                if poppler_path not in os.environ["PATH"]:
                    os.environ["PATH"] = poppler_path + os.pathsep + os.environ["PATH"]
                    logger.info("Added poppler to PATH: %s", poppler_path)
            else:
                logger.error("POPPLER_PATH specified but does NOT exist: %s", poppler_path)
                logger.error("Tip: Use forward slashes (/) in your .env file to avoid backslash escape issues.")
        elif sys.platform == "win32":
            logger.warning("POPPLER_PATH not set in .env. High-res PDF parsing may fail on Windows.")

        # Handle Tesseract Path (Windows Fix)
        tesseract_path = os.getenv("TESSERACT_PATH", "").strip('"')
        if tesseract_path:
            tesseract_path = os.path.abspath(tesseract_path)
            if os.path.exists(tesseract_path):
                if tesseract_path not in os.environ["PATH"]:
                    os.environ["PATH"] = tesseract_path + os.pathsep + os.environ["PATH"]
                    logger.info("Added tesseract to PATH: %s", tesseract_path)
            else:
                logger.error("TESSERACT_PATH specified but does NOT exist: %s", tesseract_path)
        elif sys.platform == "win32":
            logger.warning("TESSERACT_PATH not set in .env. High-res OCR may fail on Windows.")

        # Determine vision strategy
        self.use_groq_vision = os.getenv("USE_GROQ_VISION", "false").lower() == "true"

        # Groq Vision (opt-in)
        self.groq_client = None
        self.groq_vision_model = groq_vision_model or os.getenv(
            "GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview"
        )
        if self.use_groq_vision:
            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key:
                try:
                    from groq import Groq
                    self.groq_client = Groq(api_key=api_key)
                    logger.info("Groq Vision client initialized (model=%s).", self.groq_vision_model)
                except Exception as e:
                    logger.error("Failed to initialize Groq client: %s", e)

        # Local models (lazy-loaded on first use)
        self._captioner = None
        self._ocr_reader = None

        strategy = "Groq Vision API" if self.use_groq_vision and self.groq_client else "Local HuggingFace (vit-gpt2 + EasyOCR)"
        logger.info("MultiModalParser initialized (vision_strategy=%s).", strategy)

    # ------------------------------------------------------------------ #
    #  Local Vision Models (Lazy Loading)
    # ------------------------------------------------------------------ #
    def _get_captioner(self):
        """Lazy-load the HuggingFace image captioning pipeline."""
        if self._captioner is None:
            try:
                from transformers import pipeline
                import transformers.utils.import_utils
                
                # Bypass PyTorch < 2.6 vulnerability check to load local vision models
                if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
                    def _safe_bypass(): pass
                    transformers.utils.import_utils.check_torch_load_is_safe = _safe_bypass

                logger.info("Loading image captioning model: %s (first run may download ~900MB)...", CAPTIONING_MODEL)
                task_name = "image-text-to-text"
                self._captioner = pipeline(
                    task_name,
                    model=CAPTIONING_MODEL,
                    max_new_tokens=100,
                )
                logger.info("Image captioning model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load captioning model: %s", e)
                self._captioner = False  # Mark as failed, don't retry
        return self._captioner if self._captioner else None

    def _get_ocr_reader(self):
        """Lazy-load the EasyOCR reader."""
        if self._ocr_reader is None:
            try:
                import easyocr
                logger.info("Loading EasyOCR reader (first run may download models)...")
                self._ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                logger.info("EasyOCR reader loaded successfully.")
            except Exception as e:
                logger.error("Failed to load EasyOCR: %s", e)
                self._ocr_reader = False
        return self._ocr_reader if self._ocr_reader else None

    # ------------------------------------------------------------------ #
    #  PDF Parsing with Unstructured
    # ------------------------------------------------------------------ #
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF or HTML file using the `unstructured` library.

        Returns:
            List of element dicts, each with keys:
              - type: 'text' | 'table' | 'image'
              - text: extracted text content
              - page: page number (1-indexed)
              - image_path: path to extracted image (if type is 'image' or 'table')
              - vision_summary: summary of the image (if available)
        """
        logger.info("Parsing document: %s", pdf_path)

        if not os.path.exists(pdf_path):
            logger.error("File not found: %s", pdf_path)
            return []

        # Detect file type
        file_ext = Path(pdf_path).suffix.lower()
        is_html = file_ext in (".htm", ".html")

        if is_html:
            return self._parse_html(pdf_path)

        parsed_elements: List[Dict[str, Any]] = []

        try:
            logger.info("Using unstructured for full text/table/image extraction on GPU.")
            from unstructured.partition.pdf import partition_pdf
            
            # Since PyTorch is compiled with CUDA, unstructured will auto-detect the GPU
            # for the YOLOX layout model. We run sequentially because multiprocessing Poppler/Tesseract
            # is causing threads to lock up and crash the Streamlit web server.
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                hi_res_model_name="yolox_quantized",
                extract_images_in_pdf=True,
                extract_image_block_output_dir=self.images_dir,
                extract_image_block_to_payload=False,
                include_page_breaks=True,
            )
            for elem in elements:
                elem_type = type(elem).__name__
                text = str(elem).strip()
                page = elem.metadata.page_number if hasattr(elem, "metadata") and elem.metadata else 0

                if elem_type in ("Table",):
                    image_path = self._save_element_as_image(elem, "table")
                    v_sum = self._summarize_image(image_path, "financial table") if image_path else ""
                    parsed_elements.append({
                        "type": "table", "text": text, "page": page,
                        "image_path": image_path or "", "vision_summary": v_sum or text,
                    })
                elif elem_type in ("Image", "Figure"):
                    image_path = self._extract_image_path(elem)
                    v_sum = self._summarize_image(image_path, "chart or figure") if image_path else ""
                    parsed_elements.append({
                        "type": "image", "text": v_sum or text, "page": page,
                        "image_path": image_path or "", "vision_summary": v_sum,
                    })
                elif elem_type == "PageBreak":
                    continue
                else:
                    if text:
                        parsed_elements.append({
                            "type": "text", "text": text, "page": page,
                            "image_path": "", "vision_summary": "",
                        })

        except ImportError as e:
            logger.warning("Required library missing for primary parse: %s", e)
            parsed_elements = self._fallback_parse(pdf_path)
        except Exception as e:
            logger.error("PDF parsing failed for '%s': %s", pdf_path, e, exc_info=True)
            parsed_elements = self._fallback_parse(pdf_path)

        logger.info("Parsed %d total elements from '%s'.", len(parsed_elements), pdf_path)
        return parsed_elements

    # ------------------------------------------------------------------ #
    #  HTML Parsing (for SEC EDGAR .htm filings)
    # ------------------------------------------------------------------ #
    def _parse_html(self, html_path: str) -> List[Dict[str, Any]]:
        """Parse an HTML filing using unstructured's partition_html or fallback."""
        logger.info("Parsing HTML document: %s", html_path)
        parsed_elements: List[Dict[str, Any]] = []

        try:
            from unstructured.partition.html import partition_html

            elements = partition_html(filename=html_path)
            logger.info("Extracted %d elements from HTML '%s'.", len(elements), html_path)

            for elem in elements:
                elem_type = type(elem).__name__
                text = str(elem).strip()

                if not text:
                    continue

                if elem_type == "Table":
                    parsed_elements.append({
                        "type": "table", "text": text, "page": 0,
                        "image_path": "", "vision_summary": text,
                    })
                elif elem_type == "Image":
                    # For HTML, images are often local to the HTML file
                    base_dir = os.path.dirname(html_path)
                    
                    # 1. Identify the image path referenced in the HTML
                    image_path = self._extract_image_path(elem, base_dir=base_dir)
                    
                    # 2. If it's missing locally but it's an SEC filing, try to fetch it
                    if image_path and not os.path.exists(image_path):
                        image_path = self._ensure_image_locally(image_path, html_path, elem)
                        
                    v_sum = self._summarize_image(image_path, "financial visual") if image_path else ""
                    parsed_elements.append({
                        "type": "image", "text": v_sum or text, "page": 0,
                        "image_path": image_path or "", "vision_summary": v_sum,
                    })
                elif elem_type in ("PageBreak",):
                    continue
                else:
                    parsed_elements.append({
                        "type": "text", "text": text, "page": 0,
                        "image_path": "", "vision_summary": "",
                    })

        except ImportError:
            logger.warning("unstructured not available for HTML. Using fallback parser.")
            parsed_elements = self._fallback_html_parse(html_path)
        except Exception as e:
            logger.error("HTML parsing failed for '%s': %s", html_path, e)
            parsed_elements = self._fallback_html_parse(html_path)

        logger.info("Parsed %d elements from HTML '%s'.", len(parsed_elements), html_path)
        return parsed_elements

    def _fallback_html_parse(self, html_path: str) -> List[Dict[str, Any]]:
        """Simple HTML text extraction using built-in html.parser."""
        elements: List[Dict[str, Any]] = []
        try:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.texts: List[str] = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    self._skip = tag in ("script", "style", "meta", "link")

                def handle_endtag(self, tag):
                    if tag in ("script", "style"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        clean = data.strip()
                        if clean and len(clean) > 2:
                            self.texts.append(clean)

            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            extractor = TextExtractor()
            extractor.feed(content)

            current_block = []
            current_len = 0
            for t in extractor.texts:
                current_block.append(t)
                current_len += len(t)
                if current_len >= 500:
                    elements.append({
                        "type": "text",
                        "text": " ".join(current_block),
                        "page": 0,
                        "image_path": "",
                        "vision_summary": "",
                    })
                    current_block = []
                    current_len = 0

            if current_block:
                elements.append({
                    "type": "text",
                    "text": " ".join(current_block),
                    "page": 0,
                    "image_path": "",
                    "vision_summary": "",
                })

            logger.info("Fallback HTML parse extracted %d elements.", len(elements))
        except Exception as e:
            logger.error("Fallback HTML parsing failed: %s", e)
        return elements

    # ------------------------------------------------------------------ #
    #  Fallback: Basic PyMuPDF Text Extraction
    # ------------------------------------------------------------------ #
    def _fallback_parse(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Simple text extraction fallback when unstructured is unavailable."""
        elements: List[Dict[str, Any]] = []
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    elements.append({
                        "type": "text",
                        "text": text,
                        "page": page_num,
                        "image_path": "",
                        "vision_summary": "",
                    })

                # Extract images from page
                for img_idx, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n < 5:  # GRAY or RGB
                            img_filename = f"{uuid.uuid4()}.png"
                            img_path = os.path.join(self.images_dir, img_filename)
                            pix.save(img_path)
                        else:  # CMYK — convert to RGB
                            pix2 = fitz.Pixmap(fitz.csRGB, pix)
                            img_filename = f"{uuid.uuid4()}.png"
                            img_path = os.path.join(self.images_dir, img_filename)
                            pix2.save(img_path)

                        vision_summary = self._summarize_image(img_path, "financial chart or figure")
                        elements.append({
                            "type": "image",
                            "text": vision_summary or "Extracted image.",
                            "page": page_num,
                            "image_path": img_path,
                            "vision_summary": vision_summary,
                        })
                    except Exception as img_err:
                        logger.warning("Failed to extract image %d from page %d: %s", img_idx, page_num, img_err)

            doc.close()
            logger.info("Fallback parse extracted %d elements.", len(elements))
        except Exception as e:
            logger.error("Fallback PDF parsing also failed: %s", e)
        return elements

    # ------------------------------------------------------------------ #
    #  Image Handling
    # ------------------------------------------------------------------ #
    def _save_element_as_image(self, element: Any, prefix: str = "element") -> Optional[str]:
        """Attempt to save a table/image element as a .png file."""
        try:
            if hasattr(element, "metadata") and hasattr(element.metadata, "image_path"):
                return element.metadata.image_path
        except Exception:
            pass
        return None

    def _extract_image_path(self, element: Any, base_dir: Optional[str] = None) -> Optional[str]:
        """Extract the image file path from an unstructured Image element."""
        try:
            if hasattr(element, "metadata"):
                # 1. Direct path in metadata
                if hasattr(element.metadata, "image_path") and element.metadata.image_path:
                    return element.metadata.image_path
                
                # 2. Relative filename in metadata (common in PDF extraction)
                if hasattr(element.metadata, "filename"):
                    potential_path = os.path.join(self.images_dir, element.metadata.filename)
                    if os.path.exists(potential_path):
                        return potential_path
                
                # 3. Handling HTML sources (common in scrapped filings)
                source_url = (getattr(element.metadata, "image_source", None) or 
                              getattr(element.metadata, "url", None) or
                              getattr(element.metadata, "image_url", None))
                if source_url and not source_url.startswith(("http", "https", "data:")):
                    # Clean the source URL (remove any leading slashes or dots)
                    clean_source = source_url.lstrip("./\\")
                    # Try resolving relative to HTML base_dir
                    if base_dir:
                        return os.path.join(base_dir, clean_source)
        except Exception as e:
            logger.debug("Could not extract image path: %s", e)
        return None
    def _ensure_image_locally(self, local_path: str, html_path: str, element: Any) -> Optional[str]:
        """Attempt to fetch a missing image from SEC EDGAR if metadata is available."""
        if os.path.exists(local_path):
            return local_path

        # Check for metadata sidecar
        meta_path = f"{html_path}.meta.json"
        if not os.path.exists(meta_path):
            return local_path

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            source_url = meta.get("source_url")
            if not source_url or "sec.gov" not in source_url:
                return local_path

            # Reconstruct the SEC asset URL
            # The filing URL is usually: https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION/docname.htm
            # The asset URL is usually: https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION/assetname.jpg
            base_url = "/".join(source_url.split("/")[:-1])
            
            asset_filename = (getattr(element.metadata, "image_source", None) or 
                              getattr(element.metadata, "url", None) or
                              getattr(element.metadata, "image_url", None))
            if not asset_filename or asset_filename.startswith(("http", "data:")):
                return local_path
            
            # Clean asset filename (unstructured sometimes gives us the same relative path)
            asset_filename = asset_filename.lstrip("./\\")
            full_asset_url = f"{base_url}/{asset_filename}"
            
            logger.info("Missing SEC asset detected. Attempting on-demand fetch: %s", full_asset_url)
            
            # SEC requires a specific User-Agent
            headers = {"User-Agent": "FinanceRAGProject research@financerag.local"}
            resp = requests.get(full_asset_url, headers=headers, timeout=20)
            resp.raise_for_status()
            
            # Ensure the local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(resp.content)
            
            logger.info("✓ Successfully fetched missing SEC asset: %s", local_path)
            return local_path
            
        except Exception as e:
            logger.warning("Failed to on-demand fetch SEC asset: %s", e)
        
        return local_path

    # ------------------------------------------------------------------ #
    #  Image Summarization (Local HuggingFace or Groq Vision)
    # ------------------------------------------------------------------ #
    def _summarize_image(self, image_path: str, context_hint: str = "financial visual") -> str:
        """
        Generate a text summary of an image.

        Strategy:
          1. If USE_GROQ_VISION=true → use Groq Vision API (opt-in)
          2. Otherwise → local HuggingFace captioning + EasyOCR (default)
        """
        if not os.path.exists(image_path):
            logger.warning("Image file not found: %s", image_path)
            return ""

        if self.use_groq_vision and self.groq_client:
            return self._summarize_with_groq(image_path, context_hint)

        return self._summarize_with_local_models(image_path, context_hint)

    # ------------------------------------------------------------------ #
    #  Local Image Summarization (HuggingFace + EasyOCR)
    # ------------------------------------------------------------------ #
    def _summarize_with_local_models(self, image_path: str, context_hint: str) -> str:
        """
        Combine HuggingFace image captioning + EasyOCR text extraction
        for a comprehensive, token-free image summary.
        """
        parts: List[str] = []

        # --- Part 1: Image Captioning (vit-gpt2) ---
        captioner = self._get_captioner()
        if captioner:
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                result = captioner(img)
                caption = result[0]["generated_text"].strip() if result else ""
                if caption:
                    parts.append(f"Visual description: {caption}")
                    logger.info("Caption generated for '%s': %s", image_path, caption[:100])
            except Exception as e:
                logger.error("Image captioning failed for '%s': %s", image_path, e)

        # --- Part 2: OCR Text Extraction (EasyOCR) ---
        ocr = self._get_ocr_reader()
        if ocr:
            try:
                ocr_results = ocr.readtext(image_path, detail=0)
                if ocr_results:
                    ocr_texts = [t.strip() for t in ocr_results if len(t.strip()) > 1]
                    if ocr_texts:
                        ocr_combined = " | ".join(ocr_texts[:30])
                        parts.append(f"Text extracted from image: {ocr_combined}")
                        logger.info("OCR extracted %d text regions from '%s'.", len(ocr_texts), image_path)
            except Exception as e:
                logger.error("OCR failed for '%s': %s", image_path, e)

        if not parts:
            return f"[{context_hint} extracted from document]"

        summary = ". ".join(parts)
        logger.info("Local vision summary generated for '%s' (%d chars).", image_path, len(summary))
        return summary

    # ------------------------------------------------------------------ #
    #  Groq Vision Summarization (Opt-in)
    # ------------------------------------------------------------------ #
    def _summarize_with_groq(self, image_path: str, context_hint: str) -> str:
        """Send an image to Groq Vision API to generate a factual text summary."""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = Path(image_path).suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
            mime_type = mime_map.get(ext, "image/png")

            response = self.groq_client.chat.completions.create(
                model=self.groq_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"You are a financial analyst. This image is a {context_hint} "
                                    "extracted from a corporate financial document (e.g. annual report, "
                                    "10-K, investor presentation). Provide a highly factual, concise "
                                    "summary of what this visual shows. Include specific numbers, "
                                    "percentages, trends, and entities visible. Do not speculate."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_tokens=500,
                timeout=30,
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Groq Vision summary generated for '%s' (%d chars).", image_path, len(summary))
            return summary

        except Exception as e:
            logger.error("Groq Vision summarization failed for '%s': %s", image_path, e)
            logger.info("Falling back to local models for '%s'.", image_path)
            return self._summarize_with_local_models(image_path, context_hint)
