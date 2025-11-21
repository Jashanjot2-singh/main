import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

export type AnalysisResult = {
  score: number;
  strengths: string[];
  gaps: string[];
  insights: string[];
};

export async function analyzeResumeWithGemini(
  resumeText: string,
  jobDescriptionText: string
): Promise<AnalysisResult> {
  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = `You are an expert technical recruiter. Analyze the following resume against the job description and provide a detailed assessment.

Resume:
${resumeText}

Job Description:
${jobDescriptionText}

Provide your analysis in the following JSON format (ONLY return valid JSON, no markdown):
{
  "score": <number between 0-100>,
  "strengths": [<array of 3-5 key strengths as strings>],
  "gaps": [<array of 2-4 skill gaps or missing requirements as strings>],
  "insights": [<array of 2-4 key insights or recommendations as strings>]
}

Be specific and provide actionable insights. The score should reflect how well the candidate matches the job requirements.`;

    const result = await model.generateContent(prompt);
    const response = result.response;
    const text = response.text();

    // Extract JSON from response (handle potential markdown code blocks)
    let jsonText = text.trim();
    if (jsonText.startsWith('```json')) {
      jsonText = jsonText.slice(7);
    }
    if (jsonText.startsWith('```')) {
      jsonText = jsonText.slice(3);
    }
    if (jsonText.endsWith('```')) {
      jsonText = jsonText.slice(0, -3);
    }
    jsonText = jsonText.trim();

    const analysis = JSON.parse(jsonText);

    // Validate the response structure
    if (
      typeof analysis.score !== 'number' ||
      !Array.isArray(analysis.strengths) ||
      !Array.isArray(analysis.gaps) ||
      !Array.isArray(analysis.insights)
    ) {
      throw new Error('Invalid analysis format from Gemini');
    }

    return analysis;
  } catch (error) {
    console.error('Gemini analysis error:', error);
    throw new Error('Failed to analyze resume with Gemini');
  }
}

export async function answerQuestionWithGemini(
  question: string,
  context: string[],
  chatHistory: Array<{ role: string; content: string }>
): Promise<string> {
  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const contextText = context.join('\n\n---\n\n');
    const historyText = chatHistory
      .map(msg => `${msg.role.toUpperCase()}: ${msg.content}`)
      .join('\n');

    const prompt = `You are a helpful AI assistant analyzing a candidate's resume. Use the provided context to answer questions accurately.

Previous Conversation:
${historyText}

Relevant Context from Resume:
${contextText}

Current Question: ${question}

Provide a clear, concise, and accurate answer based on the context. If the information is not available in the context, say so politely. Be conversational but professional.`;

    const result = await model.generateContent(prompt);
    const response = result.response;
    return response.text();
  } catch (error) {
    console.error('Gemini chat error:', error);
    throw new Error('Failed to get answer from Gemini');
  }
}
