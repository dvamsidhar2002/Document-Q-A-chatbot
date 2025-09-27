#!/usr/bin/env python3
"""
Streamlit UI for Document Q&A Chatbot with Llama 3.1
Upload multiple documents and chat with them through a beautiful web interface
"""

import streamlit as st

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import tempfile
import requests
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import math
import time

# Document processing imports
try:
    import PyPDF2
    from docx import Document
    PDF_SUPPORT = True
    DOCX_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    DOCX_SUPPORT = False

# Optional: Enhanced text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_SUPPORT = True
    
    # Download required NLTK data
    @st.cache_resource
    def download_nltk_data():
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    download_nltk_data()
        
except ImportError:
    NLTK_SUPPORT = False

# Check for missing dependencies after page config
if not PDF_SUPPORT or not DOCX_SUPPORT:
    st.error("⚠️ Missing dependencies! Install with: pip install PyPDF2 python-docx")

class EnhancedDocumentProcessor:
    """Enhanced document processing for Streamlit"""
    
    def __init__(self):
        if NLTK_SUPPORT:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stemmer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    def load_document(self, file_content, file_name: str) -> str:
        """Load document from uploaded file"""
        file_extension = Path(file_name).suffix.lower()
        
        if file_extension == '.txt':
            return file_content.decode('utf-8')
        elif file_extension == '.pdf' and PDF_SUPPORT:
            return self._load_pdf_from_bytes(file_content)
        elif file_extension == '.docx' and DOCX_SUPPORT:
            return self._load_docx_from_bytes(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_pdf_from_bytes(self, file_content) -> str:
        """Extract text from PDF bytes"""
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        os.unlink(tmp_file.name)
        return text
    
    def _load_docx_from_bytes(self, file_content) -> str:
        """Extract text from DOCX bytes"""
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            
            doc = Document(tmp_file.name)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        os.unlink(tmp_file.name)
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text.strip())
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 10 and line.isdigit():
                continue
            if re.match(r'^[^\w]*$', line):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def smart_chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150, source_name: str = "") -> List[Dict]:
        """Enhanced chunking with semantic boundaries and source tracking"""
        text = self.preprocess_text(text)
        
        if len(text) <= chunk_size:
            return [{"text": text, "start": 0, "end": len(text), "chunk_id": 0, "source": source_name}]
        
        chunks = []
        chunk_id = 0
        
        if NLTK_SUPPORT:
            sentences = sent_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunk_end = current_start + len(current_chunk)
                    chunks.append({
                        "text": current_chunk.strip(),
                        "start": current_start,
                        "end": chunk_end,
                        "chunk_id": chunk_id,
                        "sentence_count": len(re.split(r'[.!?]+', current_chunk)),
                        "source": source_name
                    })
                    chunk_id += 1
                
                current_chunk = sentence
                current_start = text.find(sentence, current_start)
        
        if current_chunk:
            chunk_end = current_start + len(current_chunk)
            chunks.append({
                "text": current_chunk.strip(),
                "start": current_start,
                "end": chunk_end,
                "chunk_id": chunk_id,
                "sentence_count": len(re.split(r'[.!?]+', current_chunk)),
                "source": source_name
            })
        
        return chunks

class SemanticSearcher:
    """Enhanced search using TF-IDF"""
    
    def __init__(self):
        if NLTK_SUPPORT:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stemmer = None
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    def preprocess_query(self, text: str) -> List[str]:
        """Preprocess text for semantic search"""
        if NLTK_SUPPORT:
            words = word_tokenize(text.lower())
        else:
            words = re.findall(r'\b\w+\b', text.lower())
        
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                if self.stemmer:
                    word = self.stemmer.stem(word)
                processed_words.append(word)
        
        return processed_words
    
    def calculate_tfidf_scores(self, query_words: List[str], chunks: List[Dict]) -> List[Tuple[float, Dict]]:
        """Calculate TF-IDF scores for relevance ranking"""
        chunk_word_counts = []
        all_words = set()
        
        for chunk in chunks:
            words = self.preprocess_query(chunk["text"])
            word_count = Counter(words)
            chunk_word_counts.append(word_count)
            all_words.update(words)
        
        idf = {}
        total_chunks = len(chunks)
        
        for word in all_words:
            chunks_with_word = sum(1 for wc in chunk_word_counts if word in wc)
            idf[word] = math.log(total_chunks / (chunks_with_word + 1))
        
        scores = []
        for i, chunk in enumerate(chunks):
            score = 0
            word_count = chunk_word_counts[i]
            total_words = sum(word_count.values())
            
            for query_word in query_words:
                if query_word in word_count:
                    tf = word_count[query_word] / total_words
                    score += tf * idf.get(query_word, 0)
            
            if chunk.get("sentence_count", 1) > 3:
                score *= 1.1
            
            scores.append((score, chunk))
        
        return scores
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict], max_chunks: int = 3) -> List[str]:
        """Find most relevant chunks using enhanced semantic search"""
        if not chunks:
            return []
        
        query_words = self.preprocess_query(question)
        if not query_words:
            return [chunk["text"] for chunk in chunks[:max_chunks]]
        
        scored_chunks = self.calculate_tfidf_scores(query_words, chunks)
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        relevant_chunks = [chunk for score, chunk in scored_chunks if score > 0]
        
        if not relevant_chunks:
            relevant_chunks = chunks[:max_chunks]
        else:
            relevant_chunks = relevant_chunks[:max_chunks]
        
        return [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in relevant_chunks]

class OllamaClient:
    """Ollama client for Streamlit"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url
        self.model = model
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except:
            return []
    
    def check_model(self) -> bool:
        """Check if the specified model is available"""
        available_models = self.get_available_models()
        return any(self.model in model for model in available_models)
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama"""
        full_prompt = self._build_prompt(prompt, context)
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 800
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload,
                timeout=45
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except requests.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build enhanced prompt"""
        if context:
            return f"""You are an expert document analyst. Answer questions accurately based ONLY on the provided document context.

INSTRUCTIONS:
1. Use ONLY information explicitly stated in the context
2. If information is not in the context, state "The document(s) do not contain information about [specific aspect]"
3. Be precise and cite specific details from the context
4. Provide comprehensive answers when information is available
5. When multiple documents are involved, mention which document contains the information

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER based on the document context:"""
        else:
            return question

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🤖 Document Q&A Chatbot")
    st.markdown("Upload your documents and start asking questions powered by Llama 3.1!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama settings
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        
        # Check Ollama connection
        ollama_client = OllamaClient(base_url=ollama_url)
        
        if ollama_client.is_available():
            st.success("✅ Ollama is running")
            
            # Model selection
            available_models = ollama_client.get_available_models()
            if available_models:
                selected_model = st.selectbox(
                    "Select Model", 
                    available_models,
                    index=0 if available_models else 0
                )
                ollama_client.model = selected_model
            else:
                st.warning("No models found. Please pull a model first:")
                st.code("ollama pull llama3.1")
        else:
            st.error("❌ Ollama not running. Start with: `ollama serve`")
            st.stop()
        
        # Advanced settings
        st.subheader("🔧 Advanced Settings")
        max_chunks = st.slider("Max chunks to use", 1, 5, 3)
        show_debug = st.checkbox("Show debug info")
        
        # Document stats
        if st.session_state.documents_loaded and st.session_state.chatbot:
            st.subheader("📄 Documents Info")
            st.write(f"**Files:** {len(st.session_state.chatbot.document_names)}")
            for name in st.session_state.chatbot.document_names:
                st.write(f"• {name}")
            st.write(f"**Total chunks:** {len(st.session_state.chatbot.document_chunks)}")
            st.write(f"**Total text length:** {st.session_state.chatbot.total_text_length:,} chars")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files to start chatting"
        )
        
        if uploaded_files:
            # Process uploaded files
            with st.spinner("Processing documents..."):
                try:
                    # Initialize components
                    processor = EnhancedDocumentProcessor()
                    searcher = SemanticSearcher()
                    
                    all_chunks = []
                    document_names = []
                    total_text_length = 0
                    processing_results = []
                    
                    # Process each file
                    for uploaded_file in uploaded_files:
                        # Load document
                        file_content = uploaded_file.read()
                        text = processor.load_document(file_content, uploaded_file.name)
                        chunks = processor.smart_chunk_text(text, source_name=uploaded_file.name)
                        
                        # Add to collections
                        all_chunks.extend(chunks)
                        document_names.append(uploaded_file.name)
                        total_text_length += len(text)
                        
                        processing_results.append({
                            'name': uploaded_file.name,
                            'chunks': len(chunks),
                            'text_length': len(text)
                        })
                    
                    # Create chatbot instance
                    class StreamlitMultiDocChatbot:
                        def __init__(self):
                            self.processor = processor
                            self.searcher = searcher
                            self.ollama = ollama_client
                            self.document_chunks = all_chunks
                            self.document_names = document_names
                            self.total_text_length = total_text_length
                            self.processing_results = processing_results
                        
                        def answer_question(self, question: str, max_chunks: int = 3):
                            relevant_chunks = self.searcher.find_relevant_chunks(
                                question, self.document_chunks, max_chunks
                            )
                            context = "\n\n".join(relevant_chunks)
                            answer = self.ollama.generate_response(question, context)
                            return answer, relevant_chunks
                    
                    st.session_state.chatbot = StreamlitMultiDocChatbot()
                    st.session_state.documents_loaded = True
                    st.session_state.uploaded_files = uploaded_files
                    
                    st.success(f"✅ {len(uploaded_files)} document(s) loaded successfully!")
                    
                    # Show processing summary
                    st.info(f"📊 Created {len(all_chunks)} total chunks from {total_text_length:,} characters")
                    
                    # Show individual file info
                    with st.expander("📋 Processing Details"):
                        for result in processing_results:
                            st.write(f"**{result['name']}:** {result['chunks']} chunks, {result['text_length']:,} chars")
                    
                    # Show document preview
                    with st.expander("📖 Document Previews"):
                        for uploaded_file in uploaded_files:
                            uploaded_file.seek(0)  # Reset file pointer
                            file_content = uploaded_file.read()
                            text = processor.load_document(file_content, uploaded_file.name)
                            st.subheader(uploaded_file.name)
                            st.text_area(f"First 500 characters of {uploaded_file.name}:", text[:500], height=150, key=f"preview_{uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.session_state.documents_loaded = False
    
    with col2:
        st.header("💬 Chat with Documents")
        
        if st.session_state.documents_loaded:
            # Chat interface
            
            # Display chat history
            chat_container = st.container()
            
            with chat_container:
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.write(question)
                    with st.chat_message("assistant"):
                        st.write(answer)
            
            # Question input
            question = st.chat_input("Ask a question about your documents...")
            
            if question:
                # Add user message to chat
                with chat_container:
                    with st.chat_message("user"):
                        st.write(question)
                
                # Generate response
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer, relevant_chunks = st.session_state.chatbot.answer_question(
                            question, max_chunks
                        )
                        
                        # Display answer
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.write(answer)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                        # Show debug info if enabled
                        if show_debug:
                            with st.expander("🔍 Debug Info"):
                                st.write(f"**Relevant chunks found:** {len(relevant_chunks)}")
                                for i, chunk in enumerate(relevant_chunks):
                                    # Find the source document for this chunk
                                    source = "Unknown"
                                    for doc_chunk in st.session_state.chatbot.document_chunks:
                                        if doc_chunk["text"] == chunk:
                                            source = doc_chunk.get("source", "Unknown")
                                            break
                                    
                                    st.write(f"**Chunk {i+1} (from {source}):**")
                                    st.text_area(f"Content:", chunk[:300] + "...", height=100, key=f"debug_{i}")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
            
            # Quick actions
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("📄 Get Summary"):
                    with st.spinner("Generating summary..."):
                        summary_question = "Provide a comprehensive summary of all the documents, highlighting the main topics and key points from each document."
                        summary, _ = st.session_state.chatbot.answer_question(summary_question)
                        st.session_state.chat_history.append((summary_question, summary))
                        st.rerun()
            
            with col_b:
                if st.button("🔑 Key Points"):
                    with st.spinner("Extracting key points..."):
                        key_points_question = "List the most important key points and main findings from all the documents, organized by document if applicable."
                        key_points, _ = st.session_state.chatbot.answer_question(key_points_question)
                        st.session_state.chat_history.append((key_points_question, key_points))
                        st.rerun()
            
            with col_c:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            
        else:
            st.info("👆 Please upload document(s) first to start chatting!")
            
            # Example questions
            st.subheader("💡 Example Questions")
            examples = [
                "What is the main topic of these documents?",
                "Can you summarize the key findings across all documents?",
                "What are the most important points from each document?",
                "Are there any specific recommendations in the documents?",
                "What methodologies were used across the documents?",
                "What are the conclusions from all documents?",
                "Compare the main themes between the documents",
                "Which document discusses [specific topic]?"
            ]
            
            for example in examples:
                st.write(f"• {example}")

if __name__ == "__main__":
    main()