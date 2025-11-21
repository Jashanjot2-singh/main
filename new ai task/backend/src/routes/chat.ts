import express from 'express';
import { chatWithRAG } from '../services/rag';
import { saveChatMessage, getChatHistory } from '../services/mysql';

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const {
      question,
      resumeText,
      jobDescriptionText,
      chatHistory = [],
      sessionId
    } = req.body;

    if (!question || !resumeText) {
      return res.status(400).json({
        error: 'Question and resume text are required'
      });
    }

    // Generate session ID if not provided
    const currentSessionId = sessionId || `session_${Date.now()}`;

    // Get chat history from database (last 5 messages)
    const dbHistory = await getChatHistory(currentSessionId, 5);

    // Combine with current chat history
    const fullHistory = [...dbHistory, ...chatHistory];

    // Save user question to database
    await saveChatMessage(currentSessionId, 'user', question);

    // Get answer using RAG
    const result = await chatWithRAG(
      question,
      resumeText,
      jobDescriptionText,
      fullHistory,
      currentSessionId
    );

    // Save assistant response to database
    await saveChatMessage(currentSessionId, 'assistant', result.answer);

    res.json({
      answer: result.answer,
      context: result.context,
      sessionId: currentSessionId
    });

  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({
      error: 'Failed to process chat request',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
