# 🤖 Document Q&A Chatbot (Streamlit + Ollama)

A **Streamlit-powered chatbot** that lets you upload **multiple documents (PDF, DOCX, TXT)** and ask questions about them.  
The app uses **semantic search** to find relevant text chunks and generates precise answers using **Llama 3.1 (via Ollama)**.  

---

## ✨ Features
- 📂 **Multi-document support** — upload and query multiple files at once.  
- 🔍 **Semantic chunking & search** — smartly splits documents and ranks relevant sections.  
- 🧠 **Ollama integration** — works with local Llama 3.1 or other Ollama models.  
- 💬 **Chat-style interface** — ask questions, get answers, and keep chat history.  
- 📊 **Document insights** — shows chunks, text length, and per-document processing details.  
- ⚡ **Quick actions** — get summaries, extract key points, or clear the chat in one click.  
- 🛠️ **Debug mode** — see which chunks were used for each answer.  

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/dvamsidhar2002/Document-Q-A-chatbot.git
cd '.\Streamlit App\'
```

### 2. Install dependencies
Make sure you’re using **Python 3.9+**.  
```bash
pip install -r requirements.txt
```

Typical dependencies include:
- `streamlit`
- `PyPDF2`
- `python-docx`
- `nltk`
- `requests`

(You may need to run `nltk.download('punkt')` and `nltk.download('stopwords')` the first time.)

### 3. Install and run Ollama
Follow [Ollama’s installation guide](https://ollama.ai/download) for your OS.  
Then start Ollama:
```bash
ollama serve
```
Pull the Llama 3.1 model:
```bash
ollama pull llama3.1
```

### 4. Run the Streamlit app
```bash
streamlit run enhanced_streamlit_app.py
```

---

## 📖 Usage
1. Open the Streamlit UI in your browser (usually [http://localhost:8501](http://localhost:8501)).  
2. Upload one or more documents (`PDF`, `DOCX`, or `TXT`).  
3. Ask questions in the chat box.  
4. Use quick action buttons to:  
   - 📄 **Get Summary** of all uploaded docs  
   - 🔑 **Extract Key Points**  
   - 🗑️ **Clear Chat**  

---

## ⚙️ Configuration
In the **sidebar**, you can:
- Set **Ollama URL** (default: `http://localhost:11434`)  
- Choose which **Ollama model** to use  
- Adjust **max chunks** for context (default: 3)  
- Enable **debug info** to inspect retrieved chunks  

---

## 🛡️ Limitations
- Uses **TF-IDF semantic search** (not embeddings yet).  
- Documents are stored only in memory (no persistence between restarts).  
- Works best with English text (NLTK preprocessing enabled).  

---

## 🔮 Future Enhancements
- [ ] Add vector embeddings for more accurate semantic search.  
- [ ] Support long-term document storage (SQLite, FAISS, or ChromaDB).  
- [ ] Add support for highlighting answer sources in text.  
- [ ] Deployable via Docker for easier setup.  

---
