
from unstructured.partition.html import partition_html

html_path = r"e:\FINANCIAL ASSISTANT\finance_rag_project\data\raw_pdfs\JNJ_20250213_10-K.htm"

try:
    elements = partition_html(filename=html_path)
    for e in elements:
        t_name = type(e).__name__
        if "Image" in t_name or "image" in t_name.lower():
            print(f"Found match: {t_name}")
except Exception as e:
    print(f"Error: {e}")
