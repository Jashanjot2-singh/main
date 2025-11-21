import express from 'express';
import multer from 'multer';
import { analyzeResumeWithGemini } from '../services/gemini';
import { parsePDF } from '../utils/pdfParser';
import { storeDocumentEmbeddings } from '../services/rag';

const router = express.Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['application/pdf', 'text/plain'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF and TXT files are allowed.'));
    }
  },
});

router.post('/', upload.fields([
  { name: 'resume', maxCount: 1 },
  { name: 'jobDescription', maxCount: 1 }
]), async (req, res) => {
  try {
    const files = req.files as { [fieldname: string]: Express.Multer.File[] };

    if (!files.resume || !files.jobDescription) {
      return res.status(400).json({
        error: 'Both resume and job description files are required'
      });
    }

    const resumeFile = files.resume[0];
    const jdFile = files.jobDescription[0];

    // Parse documents
    let resumeText: string;
    let jdText: string;

    if (resumeFile.mimetype === 'application/pdf') {
      resumeText = await parsePDF(resumeFile.buffer);
    } else {
      resumeText = resumeFile.buffer.toString('utf-8');
    }

    if (jdFile.mimetype === 'application/pdf') {
      jdText = await parsePDF(jdFile.buffer);
    } else {
      jdText = jdFile.buffer.toString('utf-8');
    }

    // Analyze with Gemini
    const analysis = await analyzeResumeWithGemini(resumeText, jdText);

    // Store embeddings for RAG
    const sessionId = `session_${Date.now()}`;
    await storeDocumentEmbeddings(sessionId, resumeText, jdText);

    res.json({
      ...analysis,
      resumeText,
      jobDescriptionText: jdText,
      sessionId
    });

  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({
      error: 'Failed to analyze documents',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
