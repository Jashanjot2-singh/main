import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

const embeddingModel = genAI.getGenerativeModel({
  model: "gemini-embedding-001"  // 768 dimensions
});

export async function generateEmbedding(text: string): Promise<number[]> {
  try {
    const result = await embeddingModel.embedContent(text);

    if (!result || !result.embedding || !result.embedding.values) {
      console.error("❌ Gemini returned empty embedding");
      return [];
    }

    console.log("Embedding size:", result.embedding.values.length);

    return result.embedding.values; // 768-dim embedding
  } catch (error) {
    console.error("Gemini embedding error:", error);
    throw new Error("Failed to generate embedding using Gemini");
  }
}

export async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  try {
    const embeddings = await Promise.all(
      texts.map(text => generateEmbedding(text))
    );

    return embeddings;
  } catch (error) {
    console.error("Batch embedding error:", error);
    throw new Error("Failed to generate embeddings with Gemini");
  }
}

export function chunkText(
  text: string,
  chunkSize = 500,
  overlap = 50
): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];

  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    if (chunk.trim()) chunks.push(chunk.trim());
  }

  return chunks;
}
