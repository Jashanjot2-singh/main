# 🤖 Agentic RAG Chatbot with Model Context Protocol (MCP)

## 📋 Overview

This project implements an advanced **Agentic Retrieval-Augmented Generation (RAG) chatbot** that processes multiple document formats and answers user questions using a sophisticated multi-agent architecture. The system leverages **Model Context Protocol (MCP)** for inter-agent communication and provides a user-friendly Streamlit interface.

## 🏗️ Architecture

### Agent-Based Design
The system consists of four specialized agents that communicate via MCP:

1. **🎯 CoordinatorAgent** - Orchestrates the entire pipeline and manages session states
2. **📄 IngestionAgent** - Handles document parsing and preprocessing
3. **🔍 RetrievalAgent** - Manages embeddings, vector storage, and semantic search
4. **💬 LLMResponseAgent** - Generates contextual responses using Large Language Models

### Model Context Protocol (MCP)
- **Structured Communication**: All agents communicate using standardized MCP messages
- **Message Tracing**: Each workflow is tracked with unique trace IDs
- **Asynchronous Processing**: Non-blocking message passing between agents
- **Error Handling**: Comprehensive error propagation and handling

## 🚀 Features

### Document Support
- **PDF**: Full text extraction and processing
- **DOCX**: Microsoft Word document processing
- **TXT/Markdown**: Plain text file support
- **CSV**: Structured data processing with row-by-row indexing
- **PPTX**: PowerPoint presentation text extraction
- **Images**: OCR-powered text extraction (PNG, JPG, JPEG)

### Advanced Capabilities
- **Multi-Turn Conversations**: Maintains context across queries
- **Source Attribution**: Shows relevant document chunks for each answer
- **Vector Search**: Semantic similarity-based document retrieval
- **Real-time Processing**: Live status updates and progress tracking
- **Session Management**: Persistent document indexing per session

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: Qdrant
- **Embeddings**: Ollama (nomic-embed-text)
- **LLM**: Ollama (llama3.2:1b)
- **Document Processing**: LangChain, PyPDF2, python-docx, python-pptx
- **OCR**: Tesseract
- **Communication**: Custom MCP implementation

## 📦 Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended
- 10GB+ free disk space

### External Dependencies
1. **Ollama**: For embeddings and LLM inference
2. **Qdrant**: Vector database for document storage
3. **Tesseract OCR**: For image text extraction

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-rag-chatbot.git
cd agentic-rag-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama
```bash
# Install Ollama (https://ollama.ai/)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# Start Ollama service
ollama serve
```

### 5. Install and Setup Qdrant
```bash
# Using Docker (Recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# Follow instructions at: https://qdrant.tech/documentation/quick-start/
```

### 6. Install Tesseract OCR
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Update the path in the code accordingly
```

## 🚀 Running the Application

### 1. Start Required Services
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Qdrant (if using Docker)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Launch the Application
```bash
streamlit run app.py
```

### 3. Access the Interface
Open your browser and navigate to: `http://localhost:8501`

## 📖 Usage Guide

### 1. Document Upload
- Click "Browse files" to select documents
- Supported formats: PDF, DOCX, TXT, CSV, PPTX, PNG, JPG, JPEG
- Wait for processing completion (indicated by green checkmark)

### 2. Ask Questions
- Enter your question in the text input field
- Click "Ask Question" to generate response
- Response generation may take up to 10 minutes for complex queries

### 3. View Results
- Review the generated answer
- Expand "Source Context" to see relevant document chunks
- Previous queries are saved in the session

### 4. System Monitoring
- Check system status in the sidebar
- View MCP message history for debugging
- Monitor session information and processing stages

## 🔧 Configuration

### Model Configuration
Edit the following parameters in the code:

```python
# LLM Model
"model": "llama3.2:1b"  # Change to your preferred model

# Embedding Model
"model": "nomic-embed-text"  # Change to your preferred embedding model

# Chunk Size
chunk_size=500  # Adjust based on your documents

# Retrieval Limit
limit=5  # Number of context chunks to retrieve
```

### Service Endpoints
```python
# Ollama API
OLLAMA_URL = "http://localhost:11434"

# Qdrant Database
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
```

## 📊 System Architecture Flow

```
User Upload → CoordinatorAgent → IngestionAgent → RetrievalAgent
                    ↓
User Query → CoordinatorAgent → RetrievalAgent → LLMResponseAgent
                    ↓
              Response Display
```

### MCP Message Flow Example
```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent", 
  "type": "RETRIEVAL_RESULT",
  "trace_id": "rag-457",
  "payload": {
    "retrieved_context": ["Document content...", "More content..."],
    "query": "What are the key findings?"
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`

2. **Qdrant Connection Error**
   - Verify Qdrant is running on port 6333
   - Check Docker container status: `docker ps`

3. **OCR Not Working**
   - Verify Tesseract installation
   - Update the tesseract path in the code for your OS

4. **Long Response Times**
   - Normal for complex documents (up to 10 minutes)
   - Consider using smaller models for faster responses

5. **Memory Issues**
   - Reduce chunk size or batch size
   - Use smaller embedding models

## 🔍 API Endpoints

### Ollama Endpoints
- **Generate**: `POST /api/generate`
- **Embeddings**: `POST /api/embeddings`
- **Models**: `GET /api/tags`

### Qdrant Endpoints
- **Collections**: `GET /collections`
- **Search**: `POST /collections/{collection}/points/search`
- **Upsert**: `PUT /collections/{collection}/points`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for document processing utilities
- **Streamlit** for the intuitive web interface
- **Qdrant** for efficient vector storage
- **Ollama** for local LLM inference
- **Tesseract** for OCR capabilities

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the system logs for error details

---

**Built with ❤️ using Agentic Architecture and Model Context Protocol**