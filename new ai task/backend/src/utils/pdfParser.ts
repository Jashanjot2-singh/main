import pdf from 'pdf-parse';

export async function parsePDF(buffer: Buffer): Promise<string> {
  try {
    const data = await pdf(buffer);

    // Extract text from PDF
    let text = data.text;

    // Clean up the text
    text = text
      .replace(/\r\n/g, '\n') // Normalize line endings
      .replace(/\n{3,}/g, '\n\n') // Remove excessive line breaks
      .trim();

    if (!text || text.length < 50) {
      throw new Error('PDF appears to be empty or unreadable');
    }

    return text;
  } catch (error) {
    console.error('PDF parsing error:', error);
    throw new Error('Failed to parse PDF file');
  }
}

export function extractTextFromBuffer(buffer: Buffer, mimeType: string): Promise<string> {
  if (mimeType === 'application/pdf') {
    return parsePDF(buffer);
  } else if (mimeType === 'text/plain') {
    return Promise.resolve(buffer.toString('utf-8'));
  } else {
    throw new Error('Unsupported file type');
  }
}
