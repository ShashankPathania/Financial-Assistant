
import os
from unstructured.partition.html import partition_html
import logging

logging.basicConfig(level=logging.INFO)

html_path = r"e:\FINANCIAL ASSISTANT\finance_rag_project\data\raw_pdfs\JNJ_20250213_10-K.htm"

try:
    elements = partition_html(filename=html_path)
    images = [e for e in elements if type(e).__name__ == "Image"]
    
    if images:
        img = images[4] # Take one of the images
        print(f"Has attribute 'image_url': {hasattr(img.metadata, 'image_url')}")
        print(f"getattr(image_url): {getattr(img.metadata, 'image_url', 'NOT_FOUND')}")
        
        md_dict = img.metadata.to_dict()
        print(f"to_dict().get('image_url'): {md_dict.get('image_url')}")
        print(f"str(img).strip(): '{str(img).strip()}' (length: {len(str(img).strip())})")

except Exception as e:
    print(f"Error: {e}")
