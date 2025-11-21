# Resume Screening Backend

Backend API for AI-powered resume screening with RAG implementation.

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Install Ollama and Pull Model

```bash
# Install Ollama from https://ollama.ai
# Then pull the embedding model
ollama pull nomic-embed-text
```

### 3. Setup MySQL Database

Start XAMPP and create the database:

```sql
CREATE DATABASE resume_screening;
```

The table will be created automatically when the server starts.

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Gemini API key and database credentials.

### 5. Start the Server

```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start
```

The server will run on http://localhost:3000

## API Endpoints

### POST /api/analyze
Upload resume and job description for analysis.

**Request:**
- Content-Type: multipart/form-data
- Body: 
  - resume: File (PDF/TXT)
  - jobDescription: File (PDF/TXT)

**Response:**
```json
{
  "score": 75,
  "strengths": ["5 years React experience", "..."],
  "gaps": ["No Kubernetes experience", "..."],
  "insights": ["Strong backend skills", "..."],
  "resumeText": "...",
  "jobDescriptionText": "...",
  "sessionId": "session_1234567890"
}
```

### POST /api/chat
Ask questions about the candidate using RAG.

**Request:**
```json
{
  "question": "Does this candidate have a degree?",
  "resumeText": "...",
  "jobDescriptionText": "...",
  "chatHistory": [],
  "sessionId": "session_1234567890"
}
```

**Response:**
```json
{
  "answer": "Yes, the candidate has a BS in Computer Science...",
  "context": ["Education: BS Computer Science..."],
  "sessionId": "session_1234567890"
}
```

### GET /health
Health check endpoint.

## Architecture

### RAG Flow
1. Document uploaded → Parsed with pdf-parse
2. Text chunked into 500-word sections with 50-word overlap
3. Chunks embedded using Ollama (nomic-embed-text)
4. Embeddings stored in Qdrant vector database
5. User question → Embedded → Vector search
6. Retrieved relevant chunks → Sent to Gemini with context
7. Generated answer returned to user
8. Chat history stored in MySQL

### Tech Stack
- **Express.js**: REST API framework
- **Gemini 2.5 Flash**: LLM for analysis and chat
- **Ollama**: Local embedding generation
- **Qdrant Cloud**: Vector database
- **MySQL**: Chat history persistence
- **pdf-parse**: PDF document parsing
- **TypeScript**: Type-safe development

## Development

### Project Structure
```
backend/
├── src/
│   ├── routes/
│   │   ├── analyze.ts       # Resume analysis endpoint
│   │   └── chat.ts          # RAG chat endpoint
│   ├── services/
│   │   ├── gemini.ts        # Gemini API integration
│   │   ├── rag.ts           # RAG implementation
│   │   ├── embeddings.ts    # Ollama embeddings
│   │   ├── vectorDB.ts      # Qdrant operations
│   │   └── mysql.ts         # Database operations
│   ├── utils/
│   │   └── pdfParser.ts     # PDF parsing utility
│   └── index.ts             # Express server
├── package.json
└── tsconfig.json
```

### Error Handling
- All endpoints include try-catch error handling
- Errors are logged to console
- User-friendly error messages returned
- Fallback mechanisms for RAG failures

### Database Management
- MySQL connection pooling for performance
- Auto-create tables on startup
- Chat history limited to last 5 messages per session
- Old chat history cleanup utility included

## Testing

Use the provided sample files in `samples/` directory:
- `resume-1.pdf` - Senior Backend Developer
- `resume-2.txt` - Full Stack Engineer  
- `jd-1.pdf` - Senior Node.js Developer
- `jd-2.txt` - Backend Architect

## Troubleshooting

### Ollama Connection Issues
```bash
# Start Ollama service
ollama serve

# Verify model is installed
ollama list
```

### MySQL Connection Failed
- Ensure XAMPP MySQL is running
- Verify credentials in .env
- Check database exists

### Qdrant Connection Issues
- Verify API key and URL in .env
- Check network connectivity
- Ensure Qdrant Cloud account is active


