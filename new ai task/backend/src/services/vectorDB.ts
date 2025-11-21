import { QdrantClient } from '@qdrant/js-client-rest';
import { v4 as uuidv4 } from 'uuid';

const QDRANT_URL = "";
const QDRANT_API_KEY = "";

const client = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY,
});
export async function ensureCollection(collectionName: string, vectorSize: number = 3072) {
  try {
    // Check if collection exists
    const collections = await client.getCollections();
    const exists = collections.collections.some(c => c.name === collectionName);

    if (!exists) {
      // Create collection
      await client.createCollection(collectionName, {
        vectors: {
          size: vectorSize,
          distance: 'Cosine'
        }
      });
      console.log(`Created Qdrant collection: ${collectionName}`);
    }
  } catch (error) {
    console.error('Error ensuring collection:', error);
    throw new Error('Failed to ensure Qdrant collection exists');
  }
}

export async function storeVectors(
  collectionName: string,
  vectors: number[][],
  metadata: Array<{ text: string; source: string }>
) {
  try {
    await ensureCollection(collectionName);

    const points = vectors.map((vector, index) => ({
      id: uuidv4(),
      vector: vector,
      payload: metadata[index]
    }));

    await client.upsert(collectionName, {
      wait: true,
      points: points
    });

    console.log(`Stored ${points.length} vectors in collection: ${collectionName}`);
  } catch (error) {
    console.error('Error storing vectors:', error);
    throw new Error('Failed to store vectors in Qdrant');
  }
}

export async function searchVectors(
  collectionName: string,
  queryVector: number[],
  limit: number = 5
): Promise<Array<{ text: string; source: string; score: number }>> {
  try {
    const searchResult = await client.search(collectionName, {
      vector: queryVector,
      limit: limit,
      with_payload: true
    });

    return searchResult.map(result => ({
      text: (result.payload?.text as string) || '',
      source: (result.payload?.source as string) || '',
      score: result.score
    }));
  } catch (error) {
    console.error('Error searching vectors:', error);
    throw new Error('Failed to search vectors in Qdrant');
  }
}

export async function deleteCollection(collectionName: string) {
  try {
    await client.deleteCollection(collectionName);
    console.log(`Deleted collection: ${collectionName}`);
  } catch (error) {
    console.error('Error deleting collection:', error);
    // Don't throw error if collection doesn't exist
  }
}
