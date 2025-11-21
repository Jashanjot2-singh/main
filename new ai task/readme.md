# Resume Screening Tool with RAG

An AI-powered Resume Screening Tool that helps recruiters upload resumes and job descriptions, then get instant match scores with the ability to ask questions about candidates via chat using Retrieval Augmented Generation (RAG).

## 🎯 Features

- **Upload Interface**: Drag-and-drop support for resume and job description (PDF/TXT)
- **AI-Powered Analysis**: Instant match scoring with Gemini 2.5 Flash
- **Smart Insights**: Get strengths, gaps, and key insights about candidates
- **RAG-Powered Chat**: Ask contextual questions about the candidate using vector embeddings
- **Chat History**: MySQL database stores conversation history (returns last 5 chats)
- **Professional UI**: Clean, modern interface with real-time updates

## 🛠️ Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for blazing-fast development
- Tailwind CSS for styling
- shadcn/ui component library
- Lucide React icons

### Backend
- Node.js 18+ with Express
- Google Gemini API (gemini-2.5-flash)
- Ollama (nomic-embed-text for embeddings)
- Qdrant Cloud (vector database)
- MySQL (XAMPP hosted)
- pdf-parse for document parsing

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Node.js 18 or higher
- npm or yarn
- XAMPP (for MySQL)
- Ollama (for embeddings)
- Gemini API key

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd resume-screening-tool
```

### 2. Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will run on `http://localhost:8080`

### 3. Backend Setup

#### Install Ollama and Pull Model

```bash
# Install Ollama from https://ollama.ai
# Pull the nomic-embed-text model
ollama pull nomic-embed-text
```

#### Setup MySQL Database

1. Start XAMPP and launch MySQL
2. Create a new database:

```sql
CREATE DATABASE resume_screening;

USE resume_screening;

CREATE TABLE chat_history (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(255) NOT NULL,
  role ENUM('user', 'assistant') NOT NULL,
  message TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session_id (session_id),
  INDEX idx_created_at (created_at)
);
```

#### Configure Backend

Create a `.env` file in the `backend` directory:

```env
PORT=3000
GEMINI_API_KEY=your_gemini_api_key_here

# MySQL Configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=resume_screening

# Qdrant Configuration
QDRANT_URL=https://fad27aed-8393-4d9b-8ba9-f912f9fad6d1.us-west-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Y20vBrp8JAJDKAnKyKuEmNQ__iqKmILT3cgvHfB38WU

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
```

#### Install Backend Dependencies

```bash
cd backend
npm install
```

#### Start Backend Server

```bash
npm start
```

The backend will run on `http://localhost:3000`

## 📁 Project Structure

```
resume-screening-tool/
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   │   ├── analyze.ts       # Resume analysis endpoint
│   │   │   └── chat.ts          # RAG chat endpoint
│   │   ├── services/
│   │   │   ├── gemini.ts        # Gemini API integration
│   │   │   ├── rag.ts           # RAG implementation
│   │   │   ├── embeddings.ts    # Ollama embeddings
│   │   │   ├── vectorDB.ts      # Qdrant operations
│   │   │   └── mysql.ts         # MySQL operations
│   │   ├── utils/
│   │   │   └── pdfParser.ts     # PDF parsing utility
│   │   └── index.ts             # Express server
│   ├── package.json
│   └── tsconfig.json
├── src/
│   ├── components/
│   │   ├── UploadSection.tsx
│   │   ├── AnalysisResults.tsx
│   │   └── ChatInterface.tsx
│   ├── pages/
│   │   └── Index.tsx
│   └── main.tsx
├── package.json
└── README.md
```

## 🔄 RAG Implementation Flow

### Document Upload & Analysis
1. User uploads resume + job description (PDF/TXT)
2. Backend parses documents using pdf-parse
3. Text is sent to Gemini 2.5 Flash for initial analysis
4. Returns: Match score, strengths, gaps, insights
5. Document chunks are embedded using nomic-embed-text
6. Embeddings stored in Qdrant vector database

### Chat with RAG
1. User asks question about candidate
2. Question is converted to embedding using Ollama
3. Vector search in Qdrant retrieves relevant resume sections
4. Retrieved context + chat history (last 5) from MySQL
5. Combined context sent to Gemini for contextual answer
6. Response stored in MySQL and returned to user

## 🎨 UI Components

### Upload Section
- Drag-and-drop file upload
- Support for PDF and TXT formats
- File validation and error handling
- Loading states during analysis

### Analysis Results
- Match score with progress bar
- Color-coded scoring (Green: 75+, Yellow: 50-74, Red: <50)
- Categorized strengths, gaps, and insights
- Smooth animations

### Chat Interface
- Real-time messaging
- Context-aware responses using RAG
- Chat history display
- Loading indicators
- Example questions for guidance

## 🔑 API Endpoints

### POST /api/analyze
Upload and analyze resume against job description

**Request:**
```typescript
FormData {
  resume: File (PDF/TXT)
  jobDescription: File (PDF/TXT)
}
```

**Response:**
```typescript
{
  score: number,
  strengths: string[],
  gaps: string[],
  insights: string[],
  resumeText: string,
  jobDescriptionText: string
}
```

### POST /api/chat
Ask questions about the candidate using RAG

**Request:**
```typescript
{
  question: string,
  resumeText: string,
  jobDescriptionText: string,
  chatHistory: Array<{role: string, content: string}>
}
```

**Response:**
```typescript
{
  answer: string,
  context: string[] // Retrieved chunks from vector DB
}
```

## 🧪 Testing

### Sample Files Included
- `samples/resume-1.pdf` - Senior Backend Developer
- `samples/resume-2.txt` - Full Stack Engineer
- `samples/jd-1.pdf` - Senior Node.js Developer
- `samples/jd-2.txt` - Backend Architect

### Test Scenarios
1. **High Match (75%+)**: Upload resume-1.pdf with jd-1.pdf
2. **Medium Match (50-74%)**: Upload resume-2.txt with jd-1.pdf
3. **Chat Questions**:
   - "Does this candidate have a degree from a state university?"
   - "Can they handle backend architecture?"
   - "What's their experience with PostgreSQL?"
   - "How many years of React experience do they have?"

## 🎯 Architecture Highlights

### RAG Implementation
- **NOT just direct LLM queries** ✅
- Proper embedding → vector search → retrieval → generation flow
- Context window optimization with relevant chunks only
- Chat history management (last 5 messages)

### Vector Database
- Qdrant Cloud for production-ready vector storage
- Efficient similarity search with nomic-embed-text embeddings
- Collection per session for isolated contexts

### Database Design
- MySQL for persistent chat history
- Session-based conversation tracking
- Indexed for fast retrieval

## 🚨 Common Issues & Solutions

### Backend Connection Failed
- Ensure backend is running on port 3000
- Check CORS configuration
- Verify all environment variables are set

### Ollama Embedding Errors
- Confirm Ollama is running: `ollama serve`
- Verify model is pulled: `ollama list`
- Check Ollama URL in .env

### MySQL Connection Issues
- Start XAMPP MySQL service
- Verify database exists
- Check credentials in .env

### Qdrant Connection Errors
- Verify API key and URL
- Check network connectivity
- Ensure collection is created

## 📚 Additional Resources

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://ollama.ai/)
- [pdf-parse Documentation](https://www.npmjs.com/package/pdf-parse)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

