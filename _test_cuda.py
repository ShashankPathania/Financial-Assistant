import time
import torch
from sentence_transformers import SentenceTransformer

def test_cuda():
    print("--- PyTorch CUDA Test ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. PyTorch will use CPU.")

    print("\n--- SentenceTransformer Test ---")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading '{model_name}'...")
    
    t0 = time.time()
    model = SentenceTransformer(model_name)
    t1 = time.time()
    
    print(f"Model loaded in {t1-t0:.2f} seconds.")
    print(f"Model is using device: {model.device}")
    
    # Test embedding generation speed
    test_sentences = [f"This is test sentence number {i} for checking embedding speed on the current device." for i in range(100)]
    
    print(f"\nEncoding {len(test_sentences)} sentences...")
    t2 = time.time()
    embeddings = model.encode(test_sentences)
    t3 = time.time()
    
    print(f"Encoding took {t3-t2:.4f} seconds.")
    print("Test Complete.")

if __name__ == "__main__":
    test_cuda()
