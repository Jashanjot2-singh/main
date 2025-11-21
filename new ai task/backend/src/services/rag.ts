import { generateEmbedding, chunkText } from './embeddings';
import { storeVectors, searchVectors } from './vectorDB';
import { answerQuestionWithGemini } from './gemini';

export async function storeDocumentEmbeddings(
  sessionId: string,
  resumeText: string,
  jobDescriptionText: string
) {
  try {
    const collectionName = `resume_${sessionId}`;

    // Chunk the documents
    const resumeChunks = chunkText(resumeText, 500, 50);
    const jdChunks = chunkText(jobDescriptionText, 500, 50);

    // Prepare all chunks with metadata
    const allChunks = [
      ...resumeChunks.map(chunk => ({ text: chunk, source: 'resume' })),
      ...jdChunks.map(chunk => ({ text: chunk, source: 'job_description' }))
    ];

    // Generate embeddings for all chunks
    const embeddings = await Promise.all(
      allChunks.map(chunk => generateEmbedding(chunk.text))
    );

    // Store in vector database
    await storeVectors(collectionName, embeddings, allChunks);

    console.log(`Stored ${allChunks.length} document chunks for session ${sessionId}`);
  } catch (error) {
    console.error('Error storing document embeddings:', error);
    throw new Error('Failed to store document embeddings');
  }
}

export async function chatWithRAG(
  question: string,
  resumeText: string,
  jobDescriptionText: string,
  chatHistory: Array<{ role: string; content: string }>,
  sessionId: string
): Promise<{ answer: string; context: string[] }> {
  try {
    const collectionName = `resume_${sessionId}`;

    // Generate embedding for the question
    const questionEmbedding = await generateEmbedding(question);

    // Search for relevant context in vector database
    const searchResults = await searchVectors(collectionName, questionEmbedding, 5);

    // Extract relevant text chunks
    const relevantChunks = searchResults
      .filter(result => result.score > 0.5) // Only include relevant results
      .map(result => result.text);

    // If no relevant chunks found, fall back to using full documents
    const context = relevantChunks.length > 0
      ? relevantChunks
      : [
          resumeText.substring(0, 1000),
          jobDescriptionText.substring(0, 500)
        ];

    // Generate answer using Gemini with retrieved context
    const answer = await answerQuestionWithGemini(question, context, chatHistory);

    return {
      answer,
      context
    };
  } catch (error) {
    console.error('RAG chat error:', error);

    // Fallback: Use Gemini without RAG if vector search fails
    try {
      const fallbackContext = [
        resumeText.substring(0, 1000),
        jobDescriptionText.substring(0, 500)
      ];
      const answer = await answerQuestionWithGemini(question, fallbackContext, chatHistory);

      return {
        answer,
        context: fallbackContext
      };
    } catch (fallbackError) {
      throw new Error('Failed to generate answer');
    }
  }
}
