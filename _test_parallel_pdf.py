import os
import logging
import time
from src.ingestion.multi_modal_parser import MultiModalParser

logging.basicConfig(level=logging.INFO)

# Make sure we have a large PDF in the system to test
test_pdf = r"e:\FINANCIAL ASSISTANT\finance_rag_project\data\raw_pdfs\GOOGL_20250905_8-K.pdf"
if not os.path.exists(test_pdf):
    # Try finding any large PDF
    for root, dirs, files in os.walk(r"e:\FINANCIAL ASSISTANT\finance_rag_project\data\raw_pdfs"):
        for f in files:
            if f.endswith(".pdf"):
                test_pdf = os.path.join(root, f)
                break

print(f"Testing parallel extraction on: {test_pdf}")

parser = MultiModalParser(images_dir="data/extracted_images")

start_time = time.time()
elements = parser.parse_pdf(test_pdf)
end_time = time.time()

print("--- Extraction Complete ---")
print(f"Time Taken: {end_time - start_time:.2f} seconds")
print(f"Total Elements Found: {len(elements)}")

# Count element types
types = {}
for el in elements:
    t = el.get("type", "unknown")
    types[t] = types.get(t, 0) + 1

print(f"Extraction Types: {types}")
if types.get("table", 0) > 0 or types.get("image", 0) > 0:
    print("SUCCESS: High resolution extraction achieved!")
else:
    print("WARNING: No tables or images found. Check GPU/Unstructured configuration.")
