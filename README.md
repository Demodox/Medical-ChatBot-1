# 🩺 Medical Chatbot with FAISS Memory & LLM Integration

A **medical chatbot** built in **Python** that uses **FAISS vector storage** for memory and integrates with an **LLM (Google Gemini / any LLM API)** to provide accurate, context-aware responses to medical queries.

---

## 🚀 Features
- **Medical Q&A**: Ask medical-related questions and get AI-generated responses.
- **Vector Search with FAISS**: Efficient semantic search over stored medical data.
- **LLM Integration**: Connects with Google Gemini or other LLMs.
- **Persistent Memory**: Stores embeddings for quick future retrieval.
- **Extensible**: Easily add new datasets, PDFs, or text documents.
- **Environment Safety**: API keys stored securely in `.env` (not committed to Git).

---

## 📂 Project Structure
```plaintext
Medical-ChatBot-1/
├── data/                        # (optional) medical datasets, documents, PDFs, etc.
├── vectorStore/
│   └── db_faiss/                 # FAISS index files and metadata
├── Connect_MemoryWIthLLM.py      # Script to connect memory (FAISS) with the LLM
├── medibot.py                    # Main chatbot application / entry point
├── memory.py                     # Memory management: embeddings, store, retrieval helpers
├── requirements.txt              # Python package dependencies
├── .env                          # Environment variables (GOOGLE_API_KEY here) - DO NOT COMMIT
├── .gitignore                    # Files & folders to ignore in Git
└── README.md                     # Project documentation (this file)

```
## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Medical-ChatBot-1.git
cd Medical-ChatBot-1

```
### 2️⃣ Create & Activate Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Add your API keys
Create a .env file in the project root and add:
```bash
GEMINI_API_KEY=your_google_gemini_api_key_here
HUGGINGFACE_API_KEY=your_hugging_face_api_key_here


```
💻 Running the ChatBot
```bash
streamlit run medibot.py 
```
🛠 Technologies Used
Python 3.x

Streamlit (Web UI)

SentenceTransformers (Vector embeddings)

FAISS / Similarity Search

Pandas & NumPy (Data processing)




