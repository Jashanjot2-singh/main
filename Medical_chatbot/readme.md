# Medical AI Chatbot with RAG Integration

A sophisticated conversational AI application designed specifically for healthcare document analysis and medical consultations. This system combines advanced document processing, Retrieval-Augmented Generation (RAG), and medical domain expertise to provide accurate, contextual responses to healthcare-related queries.

## Project Overview

This medical chatbot enables users to:

- Upload and analyze various healthcare documents (PDF, DOCX, TXT, Excel, Images)
- Ask questions about medical symptoms, medications, and treatments
- Process voice messages for hands-free interaction
- Maintain conversation context for personalized healthcare guidance
- Store and retrieve document information using vector embeddings

## Core Features

### Multi-Modal Document Processing
- **Supported Formats**: PDF, DOCX, TXT, XLSX, XLS, PNG, JPG, JPEG, GIF, BMP, TIFF
- **Medical Document Types**: Lab reports, prescriptions, discharge summaries, clinical notes
- **Text Extraction**: Advanced preprocessing for medical terminology and structured data
- **Image Analysis**: Direct processing of medical images using Google's Gemini AI

### RAG Implementation
- **Vector Storage**: Qdrant vector database for efficient document embeddings
- **Embeddings Model**: Nomic-Embed-Text for semantic understanding
- **Retrieval Strategy**: Context-aware document retrieval with similarity search
- **Full Document Storage**: Complete document content preservation alongside chunked vectors

### Intelligent Chat Interface
- **Real-time Conversations**: Instant responses with medical context awareness
- **Multi-language Support**: Hindi and English language processing
- **Voice Integration**: Audio message transcription and processing
- **Context Maintenance**: Conversation history for follow-up questions
- **Medical Disclaimers**: Appropriate safety warnings and professional consultation recommendations

### Healthcare-Specific Features
- **Medical Terminology**: Accurate handling of medical abbreviations and terms
- **Safety Protocols**: Built-in disclaimers and professional consultation guidance
- **Data Privacy**: Local storage and processing for sensitive medical information
- **Contextual Responses**: Symptom-based recommendations with appropriate precautions

## Technology Stack

### Backend
- **Framework**: Flask (Python 3.8+)
- **AI/ML**: Google Gemini 2.5-Flash for LLM capabilities
- **Vector Database**: Qdrant for document embeddings
- **Database**: MySQL for chat history and document metadata
- **Embedding Model**: Ollama Nomic-Embed-Text

### Frontend
- **Technology**: Vanilla JavaScript with modern ES6+ features
- **UI/UX**: Responsive design with dark theme
- **File Handling**: Drag-and-drop interface with progress indicators
- **Voice Processing**: Web Speech API integration

### Document Processing
- **PDF**: PyPDF2 for text extraction
- **Word Documents**: python-docx for DOCX processing
- **Spreadsheets**: pandas and openpyxl for Excel files
- **Images**: Google Gemini Vision for medical image analysis

### Dependencies
```
txtflask==2.3.3
google-generativeai==0.3.2
python-dotenv==1.0.0
mysql-connector-python==8.1.0
qdrant-client==1.6.1
requests==2.31.0
PyPDF2==3.0.1
python-docx==0.8.11
pandas==2.0.3
openpyxl==3.1.2
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MySQL Server (XAMPP recommended for development)
- Qdrant Server
- Ollama with Nomic-Embed-Text model

### 1. Clone Repository
```
git clone <repository-url>
cd medical-ai-chatbot
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Database Setup
```
# Start XAMPP MySQL service
# The application will automatically create the database and tables
```
### 4. Qdrant Setup
```
# Install and run Qdrant locally
```
```
docker run -p 6333:6333 qdrant/qdrant
```
### 5. Ollama Setup
```
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
# Pull the embedding model
ollama pull nomic-embed-text
```
### 6. Environment Configuration
```
Create a .env file in the project root:
GOOGLE_API_KEY=your_gemini_api_key_here
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=
```
### 7. Run Application
```
python app.py
Access the application at http://localhost:8080
```
## Usage Guide
### Basic Chat Interaction
- Start Conversation: Type medical questions in the chat input

- Voice Messages: Click microphone icon to record voice queries

- File Upload: Use the "Upload Files" tab to analyze documents

- Context Awareness: System maintains conversation history for follow-up questions

### Document Analysis Workflow
- Upload Documents: Drag and drop or browse files

- Optional Questions: Add specific questions about the documents

- Analysis: System processes documents and provides medical insights

- Follow-up: Ask additional questions about uploaded content

## Sample Interactions
```
User: "I have fever and headache, what should I take?"
Bot: Suggests common medications with disclaimers

User: [Uploads lab report]
Bot: Analyzes report and provides interpretation

User: "मुझे बुखार है" (Hindi: "I have fever")
Bot: Responds in Hindi with appropriate medical advice
System Architecture
```
## RAG Pipeline
- Document Ingestion: Files processed and text extracted

- Embedding Generation: Nomic-Embed-Text creates semantic vectors

- Vector Storage: Qdrant stores embeddings with metadata

- Query Processing: User questions converted to embeddings

- Similarity Search: Relevant documents retrieved

- Response Generation: Gemini AI generates contextual responses

## Data Flow
```
User Input → RAG Retrieval → Document Context → Gemini AI → Response
Voice/File → Processing → Embedding → Qdrant → Context Integration
```
## Security & Privacy
- Local Processing: All document processing happens locally

- No External Data Sharing: Medical information stays within the system

- Secure Storage: MySQL and Qdrant for local data persistence

- Session Management: Automatic chat history cleanup

- Medical Disclaimers: Consistent reminders to consult healthcare professionals

## Performance Optimizations
- Chunking Strategy: Full document storage for complete context

- Caching: Embedding reuse for similar queries

- Async Processing: Non-blocking file uploads and analysis

- Memory Management: Automatic cleanup of temporary files

- Response Time: Optimized vector search for quick retrieval

## Testing & Validation
#### Sample Healthcare Documents
The system has been tested with:

- Laboratory test results

- Prescription documents

- Medical discharge summaries

- Radiology reports

- Clinical consultation notes

## Use Case Scenarios
```
"What medications were prescribed in this document?"
```
```
"Summarize the key findings from the lab report"
```
```
"What was the patient's blood pressure reading?"
```
```
"Are there any drug allergies mentioned?"
```
```
"What follow-up care was recommended?"
```
## Limitations & Considerations
- Not a Medical Professional: System provides general information only

- Requires Professional Consultation: All medical decisions should involve healthcare providers

- Document Quality: OCR accuracy depends on document clarity

- Language Support: Primary support for English and Hindi

- Internet Dependency: Requires connection for Gemini AI processing

## Future Enhancements
- Multi-document cross-referencing

- Advanced medical entity recognition

- Integration with medical databases

- Mobile application development

- Enhanced multilingual support

- DICOM medical imaging support


## Medical Disclaimer
This application is designed for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions. Never disregard professional medical advice or delay seeking treatment because of information provided by this system.

## Acknowledgments
- Google Gemini AI for advanced language processing

- Qdrant for vector database capabilities

- Ollama community for embedding models

- Healthcare professionals who provided domain expertise

```
Version: 1.0.0
Author: Jashanjot Singh
Contact: Jashanjotdhiman@gmail.com