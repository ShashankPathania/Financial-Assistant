"""
Multi-Modal PDF Parser
========================
Uses the `unstructured` library for layout-aware PDF parsing that correctly
handles text paragraphs, financial tables, and embedded charts.  Extracted
images/tables are saved as .png files and summarized via Groq Vision API.
"""

import base64
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EXTRACTED_IMAGES_DIR = "data/extracted_images"


class MultiModalParser:
    """
    Parse PDFs using the `unstructured` library, extract images/tables,
    and summarize visuals via Groq Vision API.
    """

    def __init__(
        self,
        images_dir: str = EXTRACTED_IMAGES_DIR,
        groq_vision_model: str = "",
    ):
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)

        self.groq_vision_model = groq_vision_model or os.getenv(
            "GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview"
        )

        # Initialize Groq client for vision tasks
        api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client = None
        if api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
                logger.info("Groq Vision client initialized (model=%s).", self.groq_vision_model)
            except Exception as e:
                logger.error("Failed to initialize Groq client: %s", e)
        else:
            logger.warning("GROQ_API_KEY not set — vision summarization will be skipped.")

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
              - vision_summary: Groq Vision summary of the image (if available)
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
            from unstructured.partition.pdf import partition_pdf

            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                extract_images_in_pdf=True,
                extract_image_block_output_dir=self.images_dir,
                extract_image_block_to_payload=False,
                include_page_breaks=True,
            )

            logger.info("Extracted %d elements from '%s'.", len(elements), pdf_path)

            for elem in elements:
                elem_type = type(elem).__name__
                text = str(elem).strip()
                page = elem.metadata.page_number if hasattr(elem, "metadata") and elem.metadata else 0

                if elem_type in ("Table",):
                    # Table element
                    image_path = self._save_element_as_image(elem, "table")
                    vision_summary = ""
                    if image_path:
                        vision_summary = self._summarize_image(image_path, "financial table")
                    parsed_elements.append({
                        "type": "table",
                        "text": text,
                        "page": page,
                        "image_path": image_path or "",
                        "vision_summary": vision_summary or text,
                    })

                elif elem_type in ("Image",):
                    # Image element
                    image_path = self._extract_image_path(elem)
                    vision_summary = ""
                    if image_path:
                        vision_summary = self._summarize_image(image_path, "financial chart or figure")
                    parsed_elements.append({
                        "type": "image",
                        "text": vision_summary or "Image extracted from document.",
                        "page": page,
                        "image_path": image_path or "",
                        "vision_summary": vision_summary,
                    })

                elif elem_type == "PageBreak":
                    continue  # Skip page breaks

                else:
                    # Text elements (NarrativeText, Title, ListItem, etc.)
                    if text:
                        parsed_elements.append({
                            "type": "text",
                            "text": text,
                            "page": page,
                            "image_path": "",
                            "vision_summary": "",
                        })

        except ImportError:
            logger.warning(
                "The `unstructured` library is not installed. "
                "Falling back to basic text extraction with PyMuPDF."
            )
            parsed_elements = self._fallback_parse(pdf_path)
        except Exception as e:
            logger.error("PDF parsing failed for '%s': %s", pdf_path, e, exc_info=True)
            parsed_elements = self._fallback_parse(pdf_path)

        logger.info("Parsed %d elements from '%s'.", len(parsed_elements), pdf_path)
        return parsed_elements

    # ------------------------------------------------------------------ #
    #  HTML Parsing (for SEC EDGAR .htm filings)
    # ------------------------------------------------------------------ #
    def _parse_html(self, html_path: str) -> List[Dict[str, Any]]:
        """Parse an HTML filing using unstructured's partition_html or BeautifulSoup fallback."""
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
                elif elem_type in ("PageBreak",):
                    continue
                else:
                    parsed_elements.append({
                        "type": "text", "text": text, "page": 0,
                        "image_path": "", "vision_summary": "",
                    })

        except ImportError:
            logger.warning("unstructured not available for HTML. Using BeautifulSoup fallback.")
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

            # Group into paragraphs of ~500 chars to avoid huge single elements
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
                        if pix.n < 5:  # is GRAY or RGB
                            img_filename = f"{uuid.uuid4()}.png"
                            img_path = os.path.join(self.images_dir, img_filename)
                            pix.save(img_path)
                        else:  # CMYK — convert to RGB first
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

    def _extract_image_path(self, element: Any) -> Optional[str]:
        """Extract the image file path from an unstructured Image element."""
        try:
            if hasattr(element, "metadata"):
                if hasattr(element.metadata, "image_path") and element.metadata.image_path:
                    return element.metadata.image_path
                # Check for images saved to output dir
                if hasattr(element.metadata, "filename"):
                    potential_path = os.path.join(self.images_dir, element.metadata.filename)
                    if os.path.exists(potential_path):
                        return potential_path
        except Exception as e:
            logger.debug("Could not extract image path: %s", e)
        return None

    # ------------------------------------------------------------------ #
    #  Groq Vision Summarization
    # ------------------------------------------------------------------ #
    def _summarize_image(self, image_path: str, context_hint: str = "financial visual") -> str:
        """
        Send an image to Groq Vision API to generate a factual text summary.
        """
        if not self.groq_client:
            logger.debug("Groq Vision client not available — skipping image summary.")
            return ""

        if not os.path.exists(image_path):
            logger.warning("Image file not found: %s", image_path)
            return ""

        try:
            # Read and encode image as base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine MIME type
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
            logger.info("Vision summary generated for '%s' (%d chars).", image_path, len(summary))
            return summary

        except Exception as e:
            logger.error("Groq Vision summarization failed for '%s': %s", image_path, e)
            return ""
