import chromadb
import gradio as gr
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 1. Configuration ---

# Le nom de la collection que nous avons créée dans le script précédent.
COLLECTION_NAME = "pizzaria_rag_collection"
# Le modèle d'embedding (doit être le même que celui utilisé pour la création).
EMBEDDING_MODEL = "mxbai-embed-large"
# Le modèle de LLM à utiliser pour la génération de la réponse.
LLM_MODEL = "mistral:latest"

# --- 2. Initialisation des composants LangChain ---

print("Initialisation des composants LangChain...")

# Initialise le client Ollama pour les embeddings
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialise le client ChromaDB pour se connecter à la base de données existante.
# Le chemin doit correspondre à l'endroit où la DB a été créée par module5_creation_db.py
# (qui est dans le même dossier 'code')
vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

# Crée un retriever à partir du vectorstore.
# Le retriever est responsable de la recherche des documents pertinents.
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Initialise le modèle de chat Ollama
llm = ChatOllama(model=LLM_MODEL)

# --- 3. Définition du prompt RAG ---

# Le template du prompt pour le LLM.
# Il inclut le contexte récupéré et la question de l'utilisateur.
template = """Tu es un assistant expert en pizzas pour la pizzeria "Bella Napoli". Utilise uniquement les informations du contexte ci-dessous pour répondre à la question de l'utilisateur.

Contexte :
{context}

Question : {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. Construction de la chaîne RAG avec LangChain Expression Language (LCEL) ---

# La chaîne RAG est construite en utilisant LCEL pour une meilleure lisibilité et modularité.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # Étape de recherche (Retrieval)
    | prompt                                                  # Étape d'augmentation (Augmented)
    | llm                                                     # Étape de génération (Generation)
    | StrOutputParser()                                       # Parse la sortie du LLM en chaîne de caractères
)

# --- 5. Interface Gradio
def answer_question(user_question):
    return rag_chain.invoke(user_question)

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Posez votre question sur nos pizzas ", placeholder="Quels allergènes dans la pizza Reine ?"),
    outputs=gr.Textbox(label="Réponse de l'assistant "),
    title="Assistant Pizza - Bella Napoli",
    description="Posez vos questions sur les pizzas, allergènes et compositions",
    theme="default"
)

if __name__ == "__main__":
    iface.launch()