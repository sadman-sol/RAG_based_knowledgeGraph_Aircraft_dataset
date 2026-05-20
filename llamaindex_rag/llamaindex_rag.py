import os
import json
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
os.environ["GROQ_API_KEY"] = ""

Settings.llm = Groq(model="llama-3.1-8b-instant", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

STORAGE_DIR = "C:/Users/USER/Downloads/silicon_task/rag/llamaindex_rag/storage"
DATA_PATH   = "C:/Users/USER/Downloads/silicon_task/rag/llamaindex_rag/data/Aircraft_Type_Designators.json"
# Load JSON Data
print("Loading data...")
with open(DATA_PATH, "r") as f:
    raw_data = json.load(f)

documents = []
for aircraft_code, triples_text in raw_data.items():
    lines    = [l.strip() for l in triples_text.strip().split("\n") if l.strip()]
    metadata = {"aircraft_code": aircraft_code, "triple_count": len(lines)}

    doc_text  = f"Aircraft Code: {aircraft_code}\n\n"
    doc_text += "Knowledge Graph Triples:\n"
    doc_text += "\n".join(lines)

    documents.append(Document(text=doc_text, metadata=metadata))

print(f"Loaded {len(documents)} aircraft records")

# Build or Load Index
if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded from storage")
else:
    print("Building new index (this may take a moment)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"Index built and saved to '{STORAGE_DIR}'")

# Query Engine
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",
)

# Interactive Console Loop
print("\n" + "="*60)
print("  Aircraft Knowledge RAG System")
print("  Ask anything about aircraft in the dataset.")
print("  Type 'quit' or 'exit' to stop.")
print("="*60)

while True:
    try:
        user_query = input("\nYour Question: ").strip()

        if not user_query:
            print("  Please enter a question.")
            continue

        if user_query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        print("\nThinking...")
        response = query_engine.query(user_query)

        print(f"\nAnswer:\n{response}")

        if hasattr(response, "source_nodes") and response.source_nodes:
            print("\nSources used:")
            for node in response.source_nodes:
                code  = node.metadata.get("aircraft_code", "unknown")
                score = round(node.score, 4) if node.score else "N/A"
                print(f"   - Aircraft [{code}]  similarity: {score}")

        print("\n" + "-"*60)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        continue