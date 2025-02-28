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


nltk.download('punkt_tab')

# 🔹 Initialize ChromaDB
st.set_page_config(page_title="Agentic AI Chatbot", layout="wide")
nlp = spacy.load("en_core_web_sm")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt_tab")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="knowledge")
# ✅ 1️⃣ Use SentenceTransformer for manual embeddings
sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# ✅ 2️⃣ Use LangChain-compatible embeddings for ChromaDB
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(client=chroma_client, collection_name="knowledge", embedding_function=embedding_function)


def show_stored_knowledge():
    """Fetches and displays all stored knowledge from ChromaDB."""
    try:
        # Get the total number of stored documents
        total_docs = chroma_collection.count()
        
        if total_docs == 0:
            return "No knowledge stored yet."

        # Fetch all available stored docs
        stored_docs = vector_store.similarity_search("", k=min(total_docs, 500))  # Limit to 50 docs max
        knowledge_texts = [doc.page_content for doc in stored_docs]

        formatted_knowledge = "\n\n".join([f"🔹 {text}" for text in knowledge_texts])
        return formatted_knowledge

    except Exception as e:
        return f"Error retrieving stored knowledge: {str(e)}"

def clear_knowledge():
    """Deletes all stored knowledge from ChromaDB."""
    try:
        # Get all stored document IDs
        all_ids = chroma_collection.get()["ids"]

        if not all_ids:
            return "No knowledge to clear."

        # Delete all documents by ID
        chroma_collection.delete(ids=all_ids)
        return "✅ All knowledge has been cleared!"

    except Exception as e:
        return f"Error clearing knowledge: {str(e)}"

def advanced_chunking(text, source, max_chunk_size=500, overlap=300, num_clusters=7):
    """Advanced Chunking: Improves retrieval by grouping meaningful chunks together."""

    sentences = sent_tokenize(text)
    doc = nlp(text)
    entities = {ent.text for ent in doc.ents}

    # Create a Knowledge Graph for Relationship Awareness
    G = nx.Graph()
    for ent in doc.ents:
        G.add_node(ent.text, label=ent.label_)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            G.add_edge(token.text, token.head.text)

    # Semantic Clustering
    embeddings = sentence_embedding_model.encode(sentences)
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

    # Recursive Splitting with Increased Size & Overlap
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=overlap)

    for chunk in semantic_chunks:
        split_chunks = text_splitter.split_text(chunk)
        final_chunks.extend(split_chunks)

    refined_chunks = []
    for chunk in final_chunks:
        contains_entity = any(entity in chunk for entity in entities)
        related_entities = [ent for ent in entities if ent in chunk]
        related_nodes = [node for node in G.nodes() if node in related_entities]
        related_edges = list(G.edges(related_nodes))

        metadata = {
            "source": source,
            "contains_entity": contains_entity,
            "related_entities": ", ".join(related_entities),
            "related_relationships": ", ".join([f"{e[0]} - {e[1]}" for e in related_edges]),
        }

        refined_chunks.append((source, chunk, metadata))

    return refined_chunks

# 🔹 Store Knowledge Function
def store_knowledge(text, source):
    """Stores knowledge using advanced chunking."""
    chunks = advanced_chunking(text, source)

    # Convert to Document format for ChromaDB
    docs = [Document(page_content=chunk[1], metadata=chunk[2]) for chunk in chunks]

    # Store in ChromaDB
    vector_store.add_documents(docs)
    
    # Check document count in ChromaDB
    total_docs = chroma_collection.count()
    print(f"✅ Stored {len(docs)} advanced chunks from {source} into ChromaDB.")
    print(f"📌 Total documents in ChromaDB: {total_docs}")


def retrieve_knowledge(query, k=5):
    """Fetches top relevant knowledge from ChromaDB and ensures proper context extraction."""
    try:
        total_docs = chroma_collection.count()
        k = min(k, total_docs)

        if total_docs == 0:
            return ["No knowledge stored yet."]

        results = vector_store.similarity_search(query, k=k)

        if not results:
            return ["No relevant knowledge found. Try rephrasing the query."]

        # 🔹 Extract meaningful sentences containing the query (instead of full chunks)
        retrieved_sentences = []
        for doc in results:
            text = doc.page_content
            matching_sentences = [sent.strip() for sent in text.split(". ") if query.lower() in sent.lower()]
            retrieved_sentences.extend(matching_sentences)

        if not retrieved_sentences:
            return ["No direct match found. Try using different keywords."]

        return retrieved_sentences
    
    except Exception as e:
        return [f"Error retrieving knowledge: {str(e)}"]


# 🔹 Live Web Search (Using SerpAPI)
search_tool = Tool(
    name="Google Search",
    func=SerpAPIWrapper(serpapi_api_key="b6af32d2f8344246cffaa779ed4efcb05a4eea10846b5dca117959b6a217984a").run,
    description="Searches Google for related information."
)


# 🔹 Fetch Related Information
def fetch_related_info(query):
    """Uses web search to find additional related information"""
    web_results = search_tool.run(query)  # Live web search
    return web_results

model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="cuda",
).eval()


# 🔹 Generate AI Answer (Only From Knowledge Base)
def generate_response(query):
    """Generates a detailed AI response using retrieved knowledge."""
    context = retrieve_knowledge(query, k=5)
    
    if not context or "No knowledge stored yet." in context:
        return "No relevant knowledge found in the uploaded data."

    context_text = " ".join(context)  # Combine relevant sentences

    if not context_text.strip():
        return "I couldn't generate a meaningful response. Try rephrasing your question."

    prompt = f"Based on the following knowledge, answer concisely:\n\nContext: {context_text}\n\nQuestion: {query}\n\nAnswer:"
    print(f"📌 AI Prompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones_like(input_ids).to("cuda")

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=300,  # 🔹 Increase response length
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,  # 🔹 Prevents AI from repeating words
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"✅ AI Response: {response}")  # Debugging print

    response = response.replace(prompt, "").strip()

    if len(response) < 5:
        return "I couldn't generate a meaningful response. Try rephrasing your question."

    return response


# 🔹 Main Function
def main():
    """Runs the AI Chatbot in Streamlit"""
    st.title("🤖 AI Chatbot V2 (Phi-2 + ChromaDB + Agentic AI)")

    # 📌 Section: Show Stored Knowledge
    st.subheader("📚 Stored Knowledge in ChromaDB")
    knowledge_content = show_stored_knowledge()
    st.text_area("📖 Current Knowledge Base", knowledge_content, height=200)

    # 📌 Add "Clear Knowledge" Button
    if st.button("🗑️ Clear Knowledge Base"):
        message = clear_knowledge()
        st.success(message)
        st.rerun()  # Refresh UI after deletion

    # 📌 Section: Upload Knowledge
    st.subheader("📤 Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        store_knowledge(text, uploaded_file.name)
        st.success("✅ Knowledge uploaded successfully!")


# User input with proper handling
user_input = st.chat_input("Ask anything...")



# 🔹 Move Chat History to a Collapsible Section
with st.expander("📜 View Chat History", expanded=False):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages inside expander
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if user_input and user_input.strip():  # 🔹 FIX: Ensure input is not empty
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response using only knowledge base
    answer = generate_response(user_input)

    # Fetch related information separately
    related_info = fetch_related_info(user_input)

    # Format AI response properly
    formatted_response = f"""
    **📖 Primary Answer (From Uploaded Knowledge):**  
    {answer}  

    **🔍 Related Information (From Web Search):**  
    {related_info}
    """

    # Display bot response
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    with st.chat_message("assistant"):
        st.markdown(formatted_response)


# Run Main Function
if __name__ == "__main__":
         main()
