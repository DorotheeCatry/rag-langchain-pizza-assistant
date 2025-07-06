# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from langchain_community.document_loaders import DirectoryLoader
import time

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"
# --- 1. Configuration ---

# Chargement de tous les fichiers PDF du dossier "Pizzaria/"
loader = DirectoryLoader("Pizzaria/", glob="**/*.pdf", show_progress=True)
docs = loader.load()
pdf_files = [doc.metadata["source"] for doc in docs]

# Nom de la collection dans ChromaDB
# Une collection est comme une table dans une base de données traditionnelle.
COLLECTION_NAME = "pizzaria_rag_collection"
# Nom du modèle d'embedding à utiliser avec Ollama.
# Ce modèle doit être téléchargé au préalable avec `ollama pull mxbai-embed-large`
EMBEDDING_MODEL = "mxbai-embed-large"

# --- 2. Chargement et découpage du document ---

def load_and_chunk_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Charge un fichier PDF, en extrait le texte et le découpe en morceaux (chunks).
    """
    print(f"Chargement du fichier : {file_path}")
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    print(f"Le document contient {len(text)} caractères.")

    print("Découpage du texte en chunks...")
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        # print(text[i : i + chunk_size])
        chunks.append(text[i : i + chunk_size])
    print(f"{len(chunks)} chunks ont été créés.")
    return chunks


# --- 3. Initialisation de ChromaDB et de la fonction d'embedding ---

print("Initialisation de ChromaDB...")
# Crée un client ChromaDB qui stockera les données sur le disque dans le dossier `chroma_db`
client = chromadb.PersistentClient(path="./chroma_db")

print("Initialisation de la fonction d'embedding via Ollama...")
# Crée une fonction d'embedding qui utilise le modèle spécifié via Ollama.
# C'est cette fonction qui sera appelée par ChromaDB pour convertir le texte en vecteurs.
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

# --- 4. Création de la collection et stockage des données ---

print(f"Création ou chargement de la collection : {COLLECTION_NAME}")
# Crée une nouvelle collection (ou la charge si elle existe déjà).
# La fonction d'embedding est passée à la création pour que la collection sache comment traiter les textes.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=ollama_ef
)

# Charge et découpe le PDF
# Charger et découper tous les PDF trouvés
all_chunks = []
for pdf_file in pdf_files:
    chunks = load_and_chunk_pdf(pdf_file)
    all_chunks.extend(chunks)

print("Stockage des chunks dans ChromaDB (cela peut prendre un certain temps)...")
# Ajoute les chunks à la collection.
# ChromaDB va automatiquement :
# 1. Utiliser la `ollama_ef` pour convertir chaque chunk en vecteur.
# 2. Stocker le chunk original (dans `documents`).
# 3. Stocker un identifiant unique pour chaque chunk (dans `ids`).
print("Ajout des chunks à ChromaDB par batchs...")

BATCH_SIZE = 10
nb_chunks = len(all_chunks)

for i in tqdm(range(0, nb_chunks, BATCH_SIZE)):
    batch = all_chunks[i:i + BATCH_SIZE]
    batch_ids = [f"chunk_{i + j}" for j in range(len(batch))]

    try:
        collection.add(documents=batch, ids=batch_ids)
    except Exception as e:
        print(f"⚠️ Erreur batch {i} : {e}")
        time.sleep(1)  # petit délai en cas d’échec pour éviter le spam


print("\n--- Base de données vectorielle créée avec succès ! ---")
print(f"Nombre de documents stockés : {collection.count()}")
