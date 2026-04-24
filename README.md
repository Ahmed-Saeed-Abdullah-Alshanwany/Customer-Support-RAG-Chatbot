# 🤖 Customer Support RAG Chatbot

An end-to-end, AI-powered customer support agent built with a Retrieval-Augmented Generation (RAG) architecture. This chatbot is designed to provide highly accurate, professional, and context-aware responses by querying a specialized support dataset, strictly preventing AI hallucinations.

## ✨ Key Features

* **Context-Aware Responses:** Utilizes RAG to search through historical customer support records and generate accurate solutions.
* **Zero Hallucination:** Engineered with strict prompting constraints to ensure the model *only* answers queries related to the provided support context. It politely declines out-of-domain questions (e.g., coding, general knowledge).
* **High-Speed Inference:** Powered by Llama-3.1-8b via the Groq API for lightning-fast response times.
* **Semantic Search:** Implements FAISS (Facebook AI Similarity Search) and `all-MiniLM-L6-v2` embeddings for rapid and accurate document retrieval.
* **Interactive UI:** A clean, user-friendly chat interface built with Streamlit.

## 🛠️ Technologies Used

* **LLM:** Llama-3.1-8b-instant (Groq API)
* **Vector Database:** FAISS
* **Embeddings:** `SentenceTransformer` (all-MiniLM-L6-v2)
* **Frontend:** Streamlit
* **Data Handling:** Pandas, NumPy

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ahmed-Saeed-Abdullah-Alshanwany/Customer-Support-RAG-Chatbot.git](https://github.com/Ahmed-Saeed-Abdullah-Alshanwany/Customer-Support-RAG-Chatbot.git)
   cd Customer-Support-RAG-Chatbot


2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt


3. **Set up your API Key:**
Replace the YOUR_API_KEY_HERE in the app.py file with your actual Groq API key.


4. **Run the application:**
   ```bash
   streamlit run app.py


🧠 System Architecture

1. Document Ingestion: The system loads a specialized customer support dataset.

2. Embedding Generation: Support instructions are converted into vector embeddings using Sentence Transformers.

3. Indexing: Embeddings are stored in a FAISS index for fast similarity search.

4. Retrieval: When a user asks a question, the system queries the FAISS index to retrieve the top 3 most relevant historical records.

5. Generation: The retrieved context and the user's question are sent to the Llama 3 model with strict system prompts to generate a formal, context-bound response.