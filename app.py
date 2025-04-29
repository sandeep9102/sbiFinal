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
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")  # Can be configured via env

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in the environment variables.")

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

# MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["sbi_assistant"]
chat_sessions = db["chat_sessions"]
feedback_collection = db["feedback"]  # To collect user feedback
metrics_collection = db["metrics"]     # To store usage metrics

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
faiss_index = None
text_chunks = []
document_metadata = {}  # Store metadata about documents

# SBI-specific categories - can be extended
SBI_CATEGORIES = [
    "Account Services",
    "Loan Information",
    "Credit Cards",
    "Digital Banking",
    "Investment Products",
    "Regulatory Information",
    "Customer Support"
]

def load_documents():
    """Load all PDF documents from the directory."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        
        if not os.path.isdir(PDF_DIRECTORY):
            logger.error(f"PDF directory path is not valid: {PDF_DIRECTORY}")
            return []
            
        if not os.listdir(PDF_DIRECTORY):
            logger.warning(f"No files found in PDF directory: {PDF_DIRECTORY}")
            return []
            
        loader = DirectoryLoader(PDF_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {PDF_DIRECTORY}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def chunk_documents(documents):
    """Split documents into text chunks for processing."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create metadata mapping for easier lookup
        for i, chunk in enumerate(chunks):
            # Store document metadata for later reference
            document_metadata[i] = {
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0)
            }
            
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return []

def get_embeddings(texts):
    """Generate embeddings for text chunks."""
    embeddings = []
    for i, doc in enumerate(texts):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=doc.page_content,
                task_type="retrieval_document"
            )
            if response and "embedding" in response:
                embeddings.append(response["embedding"])
            else:
                logger.warning(f"No embedding returned for chunk {i}")
        except Exception as e:
            logger.error(f"Error generating embedding for text chunk {i}: {e}")
            continue
    return embeddings

def initialize_faiss():
    """Initialize FAISS index with document embeddings."""
    global faiss_index, text_chunks
    
    # Load all documents from directory
    documents = load_documents()
    if not documents:
        logger.warning("No documents loaded. FAISS index initialization skipped.")
        return
        
    # Process documents into chunks
    text_chunks = chunk_documents(documents)
    if not text_chunks:
        logger.warning("No text chunks created. FAISS index initialization skipped.")
        return

    # Generate embeddings
    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        logger.error("No embeddings generated. FAISS index initialization failed.")
        return

    # Create FAISS index
    embeddings_np = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_np)
    logger.info(f"FAISS index initialized with {len(embeddings)} vectors of dimension {dimension}")

# Initialize FAISS on startup
initialize_faiss()

def retrieve_relevant_documents(query_embedding, n_results=5):
    """Retrieve top matching documents from FAISS index."""
    if faiss_index is None:
        logger.error("FAISS index is not initialized.")
        return ["Error: Knowledge base not initialized."]

    try:
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = faiss_index.search(query_embedding_np, n_results)
        
        # Return documents with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(text_chunks):
                metadata = document_metadata.get(idx, {})
                results.append({
                    "content": text_chunks[idx].page_content,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", 0),
                    "distance": float(distances[0][i])  # Convert to float for JSON serialization
                })
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def categorize_query(query):
    """Categorize the user query into relevant SBI service categories."""
    try:
        prompt = f"""
        Categorize the following customer query into one or more of these SBI service categories:
        {", ".join(SBI_CATEGORIES)}
        
        Query: {query}
        
        Return only the category names, separated by commas if multiple apply.
        """
        
        response = llm.invoke(prompt)
        categories = response.content.strip().split(",")
        return [category.strip() for category in categories if category.strip()]
    except Exception as e:
        logger.error(f"Error categorizing query: {e}")
        return []

def generate_response(query, context_docs, chat_history):
    """Generate AI response using LLM."""
    try:
        # Format context for the LLM
        context_text = ""
        if context_docs:
            context_text = "\n\n".join([f"Document: {doc['source']}, Page: {doc['page']}\n{doc['content']}" 
                                       for doc in context_docs])
        
        # Format chat history
        history_text = "\n".join([f"Customer: {item['query']}\nSBI Assistant: {item['response']}" 
                                 for item in chat_history]) if chat_history else ""
        
        # Get query categories
        categories = categorize_query(query)
        category_text = ", ".join(categories) if categories else "General Inquiry"
        
        prompt = f"""
        You are SBI's (State Bank of India) official customer service assistant. Your goal is to provide accurate, 
        helpful, and professional responses to customer inquiries about SBI's products and services.

        Current query category: {category_text}

        Guidelines:
        - Be courteous and professional, always addressing the customer respectfully.
        - Provide specific information about SBI's products and services based on the context provided.
        - If you don't have specific information, be honest but helpful, and suggest how the customer can get more details.
        - For account-specific queries, never ask for sensitive information like passwords, PINs, or full account numbers.
        - Direct customers to the official SBI website (sbi.co.in), mobile app, or nearest branch for transactions and personal account information.
        - Keep responses concise but thorough.
        - Include relevant SBI contact information when appropriate.

        Relevant Information from SBI Knowledge Base:
        {context_text if context_text else "No specific information available for this query."}

        Previous Conversation:
        {history_text}

        Customer's Query:
        {query}

        Your Response as SBI Assistant:
        """
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize for the inconvenience. Our system is experiencing technical difficulties. Please try again later or contact SBI customer support at 1800-11-2211."

def track_metrics(session_id, query, categories):
    """Track usage metrics for analytics."""
    try:
        metrics_collection.insert_one({
            "timestamp": datetime.now(),
            "session_id": session_id,
            "query_length": len(query),
            "categories": categories,
        })
    except Exception as e:
        logger.error(f"Error tracking metrics: {e}")

def rag_with_chat_history(query, session_id):
    """Perform retrieval-augmented generation with chat history."""
    try:
        # Embed the query
        query_response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )

        if not query_response or "embedding" not in query_response:
            logger.error("Failed to generate embedding for query")
            return "I'm having trouble processing your request. Please try again.", []

        # Get relevant documents
        query_embedding = query_response["embedding"]
        relevant_docs = retrieve_relevant_documents(query_embedding)

        # Get chat history
        session = chat_sessions.find_one({"session_id": session_id})
        chat_history = session.get("chat_history", []) if session else []

        # Categorize the query
        categories = categorize_query(query)

        # Generate response
        response = generate_response(query, relevant_docs, chat_history)

        # Update chat history
        new_message = {
            "query": query, 
            "response": response,
            "timestamp": datetime.now(),
            "categories": categories
        }
        
        chat_sessions.update_one(
            {"session_id": session_id},
            {"$push": {"chat_history": new_message}},
            upsert=True
        )

        # Track metrics
        track_metrics(session_id, query, categories)

        return response, chat_sessions.find_one({"session_id": session_id})["chat_history"]
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return "I apologize for the inconvenience. Our system is experiencing technical difficulties. Please try again later.", []

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to responses."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/chat/start", methods=["POST"])
def start_chat():
    """Create a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        chat_sessions.insert_one({
            "session_id": session_id, 
            "chat_history": [],
            "created_at": datetime.now()
        })
        return jsonify({"session_id": session_id})
    except Exception as e:
        logger.error(f"Error starting new chat session: {e}")
        return jsonify({"error": "Failed to create session"}), 500

@app.route("/chat/history/<session_id>", methods=["GET"])
def get_chat_history(session_id):
    """Retrieve chat history for a session."""
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
        logger.error(f"Error fetching chat history: {e}")
        return jsonify({"error": "Failed to retrieve chat history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Process a chat message and return a response."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        session_id = data.get("session_id", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Create a new session if none provided
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
        logger.error(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Collect user feedback about responses."""
    try:
        data = request.json
        session_id = data.get("session_id")
        rating = data.get("rating")
        comment = data.get("comment", "")
        
        if not session_id or rating is None:
            return jsonify({"error": "Session ID and rating are required"}), 400
            
        feedback_collection.insert_one({
            "session_id": session_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now()
        })
        
        return jsonify({"status": "Feedback submitted successfully"})
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({"error": "Failed to submit feedback"}), 500

@app.route("/reload", methods=["POST"])
def reload_documents():
    """Admin endpoint to reload documents and rebuild the index."""
    try:
        # Authenticate admin - in a real app, add proper authentication
        auth_key = request.headers.get("X-Admin-Key")
        if not auth_key or auth_key != os.getenv("ADMIN_KEY"):
            return jsonify({"error": "Unauthorized"}), 401
            
        initialize_faiss()
        return jsonify({"status": "Knowledge base reloaded successfully"})
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        return jsonify({"error": "Failed to reload documents"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)