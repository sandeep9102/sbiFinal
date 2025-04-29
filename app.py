from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import faiss
import numpy as np
import os
import uuid
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sbi_assistant.log"), logging.StreamHandler()]
)
logger = logging.getLogger("sbi_assistant")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./sbi_documents")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin_secret_key")  # default for testing

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in the environment variables.")

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

# MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["sbi_assistant"]
chat_sessions = db["chat_sessions"]
feedback_collection = db["feedback"]
metrics_collection = db["metrics"]

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global Variables
faiss_index = None
text_chunks = []
document_metadata = {}

SBI_CATEGORIES = [
    "Account Services", "Loan Information", "Credit Cards",
    "Digital Banking", "Investment Products", 
    "Regulatory Information", "Customer Support"
]

def load_documents():
    """Load all PDF documents from the directory."""
    try:
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        if not os.listdir(PDF_DIRECTORY):
            logger.warning(f"No files found in {PDF_DIRECTORY}")
            return []
        
        loader = DirectoryLoader(PDF_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return []

def chunk_documents(documents):
    """Split documents into smaller text chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            document_metadata[i] = {
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0)
            }
        logger.info(f"Chunked into {len(chunks)} parts.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        return []

def get_embeddings(texts):
    """Generate embeddings using Google GenAI."""
    embeddings = []
    for i, doc in enumerate(texts):
        try:
            resp = genai.embed_content(
                model="models/embedding-001",
                content=doc.page_content,
                task_type="retrieval_document"
            )
            if resp and "embedding" in resp:
                embeddings.append(resp["embedding"])
            else:
                logger.warning(f"No embedding for chunk {i}")
        except Exception as e:
            logger.error(f"Embedding error for chunk {i}: {e}")
    return embeddings

def initialize_faiss():
    """Initialize FAISS index."""
    global faiss_index, text_chunks

    documents = load_documents()
    if not documents:
        logger.warning("No documents to index.")
        return

    text_chunks = chunk_documents(documents)
    if not text_chunks:
        logger.warning("No text chunks created.")
        return

    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        logger.error("No embeddings generated.")
        return

    embeddings_np = np.array(embeddings, dtype=np.float32)
    dim = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings_np)
    faiss.write_index(faiss_index, "faiss.index")
    logger.info(f"FAISS initialized with {len(embeddings)} vectors.")

def load_faiss_index():
    """Load FAISS index if available."""
    global faiss_index
    try:
        if os.path.exists("faiss.index"):
            faiss_index = faiss.read_index("faiss.index")
            logger.info("Loaded FAISS index from file.")
        else:
            logger.warning("FAISS index file not found. Reinitializing...")
            initialize_faiss()
    except Exception as e:
        logger.error(f"Could not load FAISS: {e}")
        faiss_index = None

def retrieve_relevant_documents(query_embedding, n_results=5):
    """Retrieve top documents using FAISS."""
    if faiss_index is None:
        logger.error("FAISS not loaded.")
        return []

    try:
        query_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = faiss_index.search(query_np, n_results)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(text_chunks):
                meta = document_metadata.get(idx, {})
                results.append({
                    "content": text_chunks[idx].page_content,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", 0),
                    "distance": float(distances[0][i])
                })
        return results
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        return []

def categorize_query(query):
    """Categorize customer query."""
    try:
        prompt = f"""Categorize into: {', '.join(SBI_CATEGORIES)}\n\nQuery: {query}\n\nOnly return categories separated by commas."""
        response = llm.invoke(prompt)
        return [cat.strip() for cat in response.content.strip().split(",") if cat.strip()]
    except Exception as e:
        logger.error(f"Categorization error: {e}")
        return []

def generate_response(query, context_docs, chat_history):
    """Generate final response."""
    try:
        context = "\n\n".join([f"Document: {doc['source']} (Page {doc['page']})\n{doc['content']}" for doc in context_docs])
        history = "\n".join([f"Customer: {h['query']}\nAssistant: {h['response']}" for h in chat_history])
        categories = categorize_query(query)
        category_text = ", ".join(categories) if categories else "General Inquiry"

        prompt = f"""
        You are SBI's official assistant.
        Query Category: {category_text}
        
        Previous Chat:
        {history}

        Relevant Documents:
        {context or 'No specific information.'}

        Customer Query:
        {query}

        Answer:
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return "I'm sorry, there is an issue. Please contact SBI Customer Care."

def track_metrics(session_id, query, categories):
    """Track query analytics."""
    try:
        metrics_collection.insert_one({
            "timestamp": datetime.now(),
            "session_id": session_id,
            "query_length": len(query),
            "categories": categories,
        })
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")

def rag_with_chat_history(query, session_id):
    """Complete retrieval-augmented generation pipeline."""
    try:
        query_emb = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        if not query_emb or "embedding" not in query_emb:
            return "Error in processing query.", []

        embedding = query_emb["embedding"]
        relevant_docs = retrieve_relevant_documents(embedding)
        session = chat_sessions.find_one({"session_id": session_id})
        chat_history = session.get("chat_history", []) if session else []

        categories = categorize_query(query)
        response = generate_response(query, relevant_docs, chat_history)

        new_entry = {
            "query": query,
            "response": response,
            "timestamp": datetime.now(),
            "categories": categories
        }

        chat_sessions.update_one(
            {"session_id": session_id},
            {"$push": {"chat_history": new_entry}},
            upsert=True
        )

        track_metrics(session_id, query, categories)
        updated_session = chat_sessions.find_one({"session_id": session_id})
        return response, updated_session.get("chat_history", [])
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return "Technical issue. Please try again later.", []

@app.after_request
def add_cors_headers(response):
    """Add CORS headers."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/chat/start", methods=["POST"])
def start_chat():
    """Start new chat session."""
    try:
        session_id = str(uuid.uuid4())
        chat_sessions.insert_one({
            "session_id": session_id,
            "chat_history": [],
            "created_at": datetime.now()
        })
        return jsonify({"session_id": session_id})
    except Exception as e:
        logger.error(f"Start session error: {e}")
        return jsonify({"error": "Failed to create session"}), 500

@app.route("/chat/history/<session_id>", methods=["GET"])
def get_chat_history(session_id):
    """Get chat history."""
    try:
        session = chat_sessions.find_one({"session_id": session_id})
        if session:
            return jsonify({
                "session_id": session_id,
                "chat_history": session["chat_history"],
                "created_at": session.get("created_at", datetime.now())
            })
        return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": "Error retrieving history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "").strip()

        if not query:
            return jsonify({"error": "Query required"}), 400

        if not session_id:
            session_id = str(uuid.uuid4())
            chat_sessions.insert_one({
                "session_id": session_id,
                "chat_history": [],
                "created_at": datetime.now()
            })

        response, updated_history = rag_with_chat_history(query, session_id)
        return jsonify({
            "response": response,
            "chat_history": updated_history,
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Submit feedback."""
    try:
        data = request.json
        session_id = data.get("session_id")
        rating = data.get("rating")
        comment = data.get("comment", "")

        if not session_id or rating is None:
            return jsonify({"error": "Session ID and rating required"}), 400

        feedback_collection.insert_one({
            "session_id": session_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now()
        })

        return jsonify({"status": "Feedback submitted"})
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"error": "Failed to submit feedback"}), 500

@app.route("/reload", methods=["POST"])
def reload_documents():
    """Reload and rebuild documents."""
    try:
        auth_key = request.headers.get("X-Admin-Key")
        if auth_key != ADMIN_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        initialize_faiss()
        return jsonify({"status": "Knowledge base reloaded successfully"})
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return jsonify({"error": "Reload failed"}), 500

# Load FAISS at startup
load_faiss_index()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
