# Aircraft Knowledge RAG System

A simple Retrieval-Augmented Generation (RAG) system built with LlamaIndex and Groq to query aircraft knowledge graph data interactively from the console.

## Project Structure
```
    rag_system/
    ├── data/
    │   └── aircraft_data.json
    ├── storage/
    │   └── (auto-generated index files)
    ├── notebooks/
    │   └── rag_pipeline.ipynb
    ├── requirements.txt
    └── README.md
```
## Tech Stack

- **LlamaIndex** - RAG framework
- **Groq** - Free LLM API (llama-3.1-8b-instant)
- **HuggingFace** - Local embeddings (BAAI/bge-small-en-v1.5)
- **Python 3.9+**

## Installation

1. Clone the repository

    git clone https://github.com/your-username/rag_system.git
    cd rag_system

2. Install dependencies

    pip install llama-index-core llama-index-llms-groq llama-index-embeddings-huggingface transformers torch sentence-transformers

3. Get your free Groq API key at https://console.groq.com

4. Set your API key in rag_pipeline.ipynb Cell 3

    os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

## Data Format

The system expects a JSON file where each key is an aircraft code and the value is a string of knowledge graph triples.

    {
      "J328": "(J328, isManufacturedBy, FAIRCHILD DORNIER)\n(J328, hasModel, 328JET)\n(J328, isTypeOf, LandPlane)\n(J328, hasEngineType, Jet)\n(J328, hasEngineCount, 2)\n(J328, hasWTC, M)",
      "UL45": "(UL45, isManufacturedBy, 3XTRIM)\n(UL45, hasModel, 450 Ultra)\n(UL45, isTypeOf, LandPlane)\n(UL45, hasEngineType, Piston)\n(UL45, hasEngineCount, 1)\n(UL45, hasWTC, L)"
    }

## Usage

Run the notebook and execute all cells. The last cell starts an interactive console loop.

    ------------------------------------------------------------
    Aircraft Knowledge RAG System
    Ask anything about aircraft in the dataset.
    Type 'quit' or 'exit' to stop.
    ------------------------------------------------------------

    Your Question: Who manufactures the J328 aircraft?

    Thinking...

    Answer:
    J328 is manufactured by 328 Support Services, AVCRAFT, Fairchild Dornier, and RUAG.

    Sources used:
       - Aircraft [J328]  similarity: 0.9123

    ------------------------------------------------------------

## Sample Questions

- Who manufactures the J328 aircraft?
- What engine type does UL45 use?
- Which aircraft has WTC category M?
- How many engines does the J328 have?
- Compare J328 and UL45 engine types.
- Which aircraft uses a piston engine?

## How It Works

1. JSON data is loaded and each aircraft record is converted into a LlamaIndex Document
2. HuggingFace embeddings are generated locally for each document
3. Documents are stored in a VectorStoreIndex and persisted to disk
4. On subsequent runs the index is loaded from storage, skipping re-embedding
5. User queries are embedded and matched against the index
6. Top 3 similar documents are retrieved and passed to Groq LLM for answering

## Notes

- The index is built once and saved to the storage/ folder
- Subsequent runs load the index from disk, saving time and resources
- HuggingFace embeddings run fully locally, no API key or cost required
- Only the Groq API key is needed for LLM inference
