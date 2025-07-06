import os
from pathlib import Path
from tqdm import tqdm
from PyPDF2 import PdfReader
from slugify import slugify
import chromadb
from chromadb.utils import embedding_functions

# CONFIGURATION
RECIPE_DIR = Path("data/recipes")
ALLERGEN_PDF_DIR = Path("data/allergens")
COLLECTION_NAME = "pizzaria_rag_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

# Initialisation de ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# Détection du flag végétarien
def detect_is_veggie(text: str) -> bool:
    txt = text.lower()
    return any(word in txt for word in ["végétarien", "végétarienne", "veggie"])

# Ajout dans la collection
def add_text_to_collection(text, uid, metadata=None):
    try:
        collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[uid]
        )
    except Exception as e:
        print(f"⚠️ Erreur avec {uid} : {e}")

# INDEXATION DES RECETTES
print("📘 Indexation des recettes PDF...")
for path in tqdm(RECIPE_DIR.glob("*.pdf")):
    reader = PdfReader(str(path))
    text = "".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        continue

    uid = f"recipe_{slugify(path.stem)}"
    is_veggie = detect_is_veggie(path.stem + " " + text)
    metadata = {
        "type": "recipe",
        "file": path.name,
        "is_veggie": is_veggie
    }
    add_text_to_collection(text, uid, metadata)

# INDEXATION DES ALLERGÈNES
print("📄 Indexation des allergènes PDF (par page)...")
for path in tqdm(ALLERGEN_PDF_DIR.glob("*.pdf")):
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content and len(content.strip()) > 100:
            uid = f"allergen_{slugify(path.stem)}_p{i+1}"
            is_veggie = detect_is_veggie(content)
            metadata = {
                "type": "allergen",
                "file": path.name,
                "page": i + 1,
                "is_veggie": is_veggie
            }
            text = f"Source : {path.name}, page {i + 1}\n{content}"
            add_text_to_collection(text, uid, metadata)

print("\n✅ ChromaDB construite avec succès.")
print(f"📦 Nombre total de documents indexés : {collection.count()}")
