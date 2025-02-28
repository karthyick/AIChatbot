# Smart AI Chatbot: Knowledge-Driven Responses Using LangChain & ChromaDB

## 📌 Overview
This project is a **Smart AI Chatbot** powered by **LangChain, ChromaDB, and OPT-1.3B** for knowledge-driven responses. It allows users to upload text-based knowledge, retrieve relevant context using semantic search, and generate intelligent responses based on the stored knowledge.

## 🚀 Features
- **Knowledge Upload**: Upload text files to ChromaDB for AI-powered retrieval.
- **Advanced Chunking**: Uses **semantic clustering** and **recursive text splitting** for efficient knowledge storage.
- **Contextual Retrieval**: Retrieves relevant information based on user queries.
- **AI-Powered Responses**: Uses **Meta OPT-1.3B** for generating natural language answers.
- **Live Web Search**: Fetches additional context using **SerpAPI (Google Search)**.
- **Streamlit UI**: Interactive user-friendly interface for seamless interaction.

## 🛠️ Tech Stack
- **Python 3.11**
- **LangChain** (for vector retrieval & AI-driven responses)
- **ChromaDB** (for storing and searching knowledge)
- **SentenceTransformer** (for embedding text into vector space)
- **OPT-1.3B** (for AI response generation)
- **Streamlit** (for UI interaction)
- **NLTK & SpaCy** (for NLP processing)
- **SerpAPI** (for live web search)

## 📂 Project Structure
```
chatbot_v2/
│── chroma_db/           # ChromaDB storage
│── app.py               # Main chatbot application
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation (this file)
```

## ⚙️ Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/chatbot_v2.git
cd chatbot_v2
```
### **2️⃣ Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```
### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```
### **4️⃣ Run the Chatbot**
```sh
streamlit run app.py
```

## 🔥 Usage
1. **Upload a text file** with knowledge (e.g., articles, notes, FAQs).
2. **Ask a question**, and the chatbot will retrieve relevant knowledge.
3. **AI generates a response** based on retrieved knowledge.

## 🔧 Troubleshooting
- **Slow response time?** Ensure your GPU is being used (`torch.cuda.is_available()`).
- **Errors with ChromaDB?** Try deleting the `chroma_db/` folder and restart the app.
- **Missing dependencies?** Run `pip install -r requirements.txt` to install required libraries.

## 📜 License
This project is open-source and available under the **MIT License**.

## 🤝 Contributing
Feel free to open issues or contribute by submitting a pull request. Let's build a smarter chatbot together! 🚀

