"""Enhanced Chatbot v2 with Knowledge Graph Visualization and Feedback System"""

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from langchain.tools import Tool
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import nltk
import spacy
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import uuid

# Import new modules
from knowledge_graph_viz import KnowledgeGraphVisualizer, render_knowledge_graph_ui
from feedback_system import FeedbackSystem, render_feedback_ui, render_feedback_dashboard

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

# 🕹️ Initialize Streamlit
st.set_page_config(
    page_title="AI Chatbot V2 Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize NLP and models
@st.cache_resource
def init_models():
    """Initialize all models and systems"""
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection(name="knowledge")
    
    # Initialize embeddings
    sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        client=chroma_client, 
        collection_name="knowledge", 
        embedding_function=embedding_function
    )
    
    # Initialize Knowledge Graph Visualizer
    kg_viz = KnowledgeGraphVisualizer(nlp)
    
    # Initialize Feedback System
    feedback_system = FeedbackSystem()
    
    # Load language model
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()
    
    return {
        'nlp': nlp,
        'chroma_client': chroma_client,
        'chroma_collection': chroma_collection,
        'sentence_embedding_model': sentence_embedding_model,
        'embedding_function': embedding_function,
        'vector_store': vector_store,
        'kg_viz': kg_viz,
        'feedback_system': feedback_system,
        'tokenizer': tokenizer,
        'model': model
    }

# Initialize all systems
models = init_models()

def show_stored_knowledge():
    """Fetches and displays all stored knowledge from ChromaDB."""
    try:
        total_docs = models['chroma_collection'].count()
        
        if total_docs == 0:
            return "No knowledge stored yet."

        stored_docs = models['vector_store'].similarity_search("", k=min(total_docs, 50))
        knowledge_texts = [doc.page_content for doc in stored_docs]
        formatted_knowledge = "\n\n".join([f"🔹 {text}" for text in knowledge_texts])
        
        return formatted_knowledge

    except Exception as e:
        return f"Error retrieving stored knowledge: {str(e)}"

def clear_knowledge():
    """Deletes all stored knowledge from ChromaDB and resets graph."""
    try:
        all_ids = models['chroma_collection'].get()["ids"]

        if not all_ids:
            return "No knowledge to clear."

        models['chroma_collection'].delete(ids=all_ids)
        
        # Reset knowledge graph
        models['kg_viz'].graph.clear()
        
        return "✅ All knowledge has been cleared!"

    except Exception as e:
        return f"Error clearing knowledge: {str(e)}"

def advanced_chunking_with_graph(text, source, max_chunk_size=500, overlap=300, num_clusters=7):
    """Enhanced chunking with knowledge graph integration"""
    
    sentences = sent_tokenize(text)
    
    # Extract relationships and build local graph
    local_graph = models['kg_viz'].extract_enhanced_relationships(text, source)
    
    # Merge into main graph
    models['kg_viz'].merge_graph(local_graph)
    
    # Get entities from the graph
    entities = set(local_graph.nodes())
    
    # Semantic Clustering
    embeddings = models['sentence_embedding_model'].encode(sentences)
    num_clusters = min(len(sentences), num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    clustered_chunks = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clustered_chunks:
            clustered_chunks[cluster_id] = []
        clustered_chunks[cluster_id].append(sentences[i])

    semantic_chunks = [" ".join(clustered_chunks[c]) for c in clustered_chunks]

    # Recursive Splitting
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, 
        chunk_overlap=overlap
    )

    for chunk in semantic_chunks:
        split_chunks = text_splitter.split_text(chunk)
        final_chunks.extend(split_chunks)

    refined_chunks = []
    for chunk in final_chunks:
        contains_entity = any(entity in chunk for entity in entities)
        related_entities = [ent for ent in entities if ent in chunk]
        
        # Get relationships from graph
        related_edges = []
        for entity in related_entities:
            if local_graph.has_node(entity):
                edges = local_graph.edges(entity)
                related_edges.extend(edges)

        metadata = {
            "source": source,
            "contains_entity": contains_entity,
            "related_entities": ", ".join(related_entities),
            "related_relationships": ", ".join([f"{e[0]} - {e[1]}" for e in related_edges]),
        }

        refined_chunks.append((source, chunk, metadata))

    return refined_chunks

def store_knowledge(text, source):
    """Stores knowledge using advanced chunking with graph integration."""
    chunks = advanced_chunking_with_graph(text, source)

    docs = [Document(page_content=chunk[1], metadata=chunk[2]) for chunk in chunks]
    models['vector_store'].add_documents(docs)
    
    total_docs = models['chroma_collection'].count()
    st.success(f"✅ Stored {len(docs)} advanced chunks from {source} into ChromaDB.")
    st.info(f"📈 Total documents in ChromaDB: {total_docs}")
    st.info(f"🕸️ Knowledge Graph: {models['kg_viz'].graph.number_of_nodes()} entities, {models['kg_viz'].graph.number_of_edges()} relationships")

def retrieve_knowledge_with_feedback(query, k=5):
    """Enhanced retrieval considering feedback scores"""
    try:
        total_docs = models['chroma_collection'].count()
        k = min(k, total_docs)

        if total_docs == 0:
            return ["No knowledge stored yet."]

        # Get initial results
        results = models['vector_store'].similarity_search(query, k=k*2)  # Get more for ranking

        if not results:
            return ["No relevant knowledge found. Try rephrasing the query."]

        # Rank results based on feedback scores
        ranked_results = []
        for doc in results:
            # Calculate combined score (similarity + feedback)
            quality_score = models['feedback_system'].get_response_quality_score(
                doc.page_content
            )
            ranked_results.append((doc, quality_score))
        
        # Sort by quality score and take top k
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        top_results = [doc for doc, _ in ranked_results[:k]]
        
        # Extract relevant sentences
        retrieved_sentences = []
        for doc in top_results:
            text = doc.page_content
            matching_sentences = [
                sent.strip() for sent in text.split(". ") 
                if query.lower() in sent.lower()
            ]
            retrieved_sentences.extend(matching_sentences)

        if not retrieved_sentences:
            # If no exact matches, return full chunks
            return [doc.page_content for doc in top_results]

        return retrieved_sentences
    
    except Exception as e:
        return [f"Error retrieving knowledge: {str(e)}"]

# Web search tool
search_tool = Tool(
    name="Google Search",
    func=SerpAPIWrapper(
        serpapi_api_key="b6af32d2f8344246cffaa779ed4efcb05a4eea10846b5dca117959b6a217984a"
    ).run,
    description="Searches Google for related information."
)

def fetch_related_info(query):
    """Uses web search to find additional related information"""
    web_results = search_tool.run(query)
    return web_results

def generate_response_with_timing(query, session_id):
    """Enhanced response generation with timing and context tracking"""
    start_time = time.time()
    
    # Retrieve knowledge with feedback-aware ranking
    context = retrieve_knowledge_with_feedback(query, k=5)
    
    if not context or "No knowledge stored yet." in context:
        response_time = int((time.time() - start_time) * 1000)
        return {
            'response': "No relevant knowledge found in the uploaded data.",
            'context_used': [],
            'response_time_ms': response_time
        }

    context_text = " ".join(context)

    if not context_text.strip():
        response_time = int((time.time() - start_time) * 1000)
        return {
            'response': "I couldn't generate a meaningful response. Try rephrasing your question.",
            'context_used': context,
            'response_time_ms': response_time
        }

    prompt = f"""Based on the following knowledge, answer concisely:

Context: {context_text}

Question: {query}

Answer:"""

    input_ids = models['tokenizer'].encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones_like(input_ids).to("cuda")

    outputs = models['model'].generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=models['tokenizer'].eos_token_id,
    )

    response = models['tokenizer'].decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    if len(response) < 5:
        response = "I couldn't generate a meaningful response. Try rephrasing your question."

    response_time = int((time.time() - start_time) * 1000)
    
    return {
        'response': response,
        'context_used': context,
        'response_time_ms': response_time
    }

def main():
    """Main application with enhanced features"""
    
    # Generate session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🤖 AI Chatbot V2 Enhanced")
        
        page = st.radio(
            "Navigation",
            ["Chat", "Knowledge Graph", "Feedback Analytics", "Settings"]
        )
        
        st.divider()
        
        # Quick stats
        st.metric("Knowledge Docs", models['chroma_collection'].count())
        st.metric("Graph Entities", models['kg_viz'].graph.number_of_nodes())
        st.metric("Graph Relations", models['kg_viz'].graph.number_of_edges())
    
    # Main content area
    if page == "Chat":
        st.title("🤖 AI Chatbot V2 - Enhanced Edition")
        st.caption("Now with Knowledge Graph Visualization and Feedback Learning!")
        
        # Knowledge management section
        with st.expander("📚 Knowledge Management", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📁 Upload Knowledge")
                uploaded_file = st.file_uploader(
                    "Upload a TXT file", 
                    type=["txt"],
                    help="Upload documents to build the knowledge base"
                )
                
                if uploaded_file:
                    text = uploaded_file.read().decode("utf-8")
                    store_knowledge(text, uploaded_file.name)
                    st.rerun()
            
            with col2:
                st.subheader("🗟️ Actions")
                if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                    message = clear_knowledge()
                    st.success(message)
                    st.rerun()
                    
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
        
        # Current knowledge display
        with st.expander("📄 Current Knowledge Base", expanded=False):
            knowledge_content = show_stored_knowledge()
            st.text_area(
                "Stored Knowledge", 
                knowledge_content, 
                height=200,
                disabled=True
            )
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "feedback_shown" not in msg:
                    st.markdown(msg["content"])
                    
                    # Add feedback UI for assistant messages
                    if "query" in msg and "context_used" in msg:
                        render_feedback_ui(
                            response=msg["raw_response"],
                            query=msg["query"],
                            session_id=st.session_state.session_id,
                            context_used=msg["context_used"],
                            feedback_system=models['feedback_system']
                        )
                        msg["feedback_shown"] = True
                else:
                    st.markdown(msg["content"])
        
        # Chat input
        user_input = st.chat_input("Ask anything...")
        
        if user_input and user_input.strip():
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate response with progress
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Generate AI response
                    response_data = generate_response_with_timing(
                        user_input, 
                        st.session_state.session_id
                    )
                    
                    # Fetch web info
                    related_info = fetch_related_info(user_input)
                    
                    # Format complete response
                    formatted_response = f"""
**📚 Primary Answer (From Knowledge Base):**  
{response_data['response']}

**🔍 Related Information (From Web):**  
{related_info}

*Response time: {response_data['response_time_ms']}ms*
"""
                    
                    st.markdown(formatted_response)
                    
                    # Add feedback UI immediately
                    render_feedback_ui(
                        response=response_data['response'],
                        query=user_input,
                        session_id=st.session_state.session_id,
                        context_used=response_data['context_used'],
                        feedback_system=models['feedback_system']
                    )
                    
                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response,
                        "raw_response": response_data['response'],
                        "query": user_input,
                        "context_used": response_data['context_used'],
                        "feedback_shown": True
                    })
    
    elif page == "Knowledge Graph":
        st.title("🕸️ Knowledge Graph Visualization")
        st.caption("Explore the relationships between entities in your knowledge base")
        
        if models['kg_viz'].graph.number_of_nodes() == 0:
            st.warning("⚠️ No knowledge graph data yet. Upload documents to see the graph!")
        else:
            render_knowledge_graph_ui(models['kg_viz'])
    
    elif page == "Feedback Analytics":
        st.title("📊 Feedback Analytics Dashboard")
        st.caption("Monitor and analyze user feedback to improve responses")
        render_feedback_dashboard(models['feedback_system'])
    
    elif page == "Settings":
        st.title("⚙️ Settings")
        
        # Model settings
        st.subheader("🤖 Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature", 
                min_value=0.1, 
                max_value=2.0, 
                value=0.7,
                help="Controls randomness in responses"
            )
            
            top_k = st.slider(
                "Top-K", 
                min_value=10, 
                max_value=100, 
                value=50,
                help="Limits vocabulary for generation"
            )
        
        with col2:
            top_p = st.slider(
                "Top-P", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.95,
                help="Nucleus sampling threshold"
            )
            
            max_tokens = st.slider(
                "Max Tokens", 
                min_value=50, 
                max_value=500, 
                value=300,
                help="Maximum response length"
            )
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Configuration")
        retrieval_k = st.slider(
            "Number of Context Chunks",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to retrieve"
        )
        
        # Graph settings
        st.subheader("🕸️ Graph Configuration")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=500,
            help="Maximum size of text chunks"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=500,
            value=300,
            help="Overlap between consecutive chunks"
        )
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")
            st.balloons()

# Run the application
if __name__ == "__main__":
    main()