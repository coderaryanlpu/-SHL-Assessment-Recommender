"""Quick test of the BM25 retriever."""
import sys
sys.path.insert(0, r"c:\Users\aryan\Desktop\shl")

from retriever import get_index, retrieve, format_for_prompt

idx = get_index()
print(f"Catalog loaded: {len(idx.catalog)} items")

queries = [
    "Java developer programming",
    "sales manager personality",
    "entry level customer service",
    "cognitive ability senior executive",
]

for q in queries:
    hits = retrieve(q, top_k=3)
    print(f"\nQuery: '{q}'")
    for h in hits:
        print(f"  [{h['test_type']}] {h['name']} (score={h['_score']})")

print("\nRetriever OK")
