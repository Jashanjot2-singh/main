import mysql from 'mysql2/promise';

let pool: mysql.Pool;

export async function initializeDatabase() {
  try {
    // Create connection pool
    pool = mysql.createPool({
      host: process.env.DB_HOST || 'localhost',
      user: process.env.DB_USER || 'root',
      password: process.env.DB_PASSWORD || '',
      database: process.env.DB_NAME || 'resume_screening',
      waitForConnections: true,
      connectionLimit: 10,
      queueLimit: 0
    });

    // Test connection
    const connection = await pool.getConnection();
    console.log('✅ MySQL connected successfully');

    // Create table if it doesn't exist
    await connection.query(`
      CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL,
        role ENUM('user', 'assistant') NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session_id (session_id),
        INDEX idx_created_at (created_at)
      )
    `);

    console.log('✅ Chat history table ready');
    connection.release();
  } catch (error) {
    console.error('❌ MySQL connection error:', error);
    throw error;
  }
}

export async function saveChatMessage(
  sessionId: string,
  role: 'user' | 'assistant',
  message: string
) {
  try {
    await pool.query(
      'INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)',
      [sessionId, role, message]
    );
  } catch (error) {
    console.error('Error saving chat message:', error);
    throw new Error('Failed to save chat message');
  }
}

export async function getChatHistory(
  sessionId: string,
  limit: number = 5
): Promise<Array<{ role: string; content: string }>> {
  try {
    const [rows] = await pool.query(
      `SELECT role, message as content
       FROM chat_history
       WHERE session_id = ?
       ORDER BY created_at DESC
       LIMIT ?`,
      [sessionId, limit]
    );

    // Reverse to get chronological order
    return (rows as Array<{ role: string; content: string }>).reverse();
  } catch (error) {
    console.error('Error getting chat history:', error);
    return [];
  }
}

export async function clearOldChatHistory(daysOld: number = 30) {
  try {
    await pool.query(
      'DELETE FROM chat_history WHERE created_at < DATE_SUB(NOW(), INTERVAL ? DAY)',
      [daysOld]
    );
    console.log(`Cleared chat history older than ${daysOld} days`);
  } catch (error) {
    console.error('Error clearing old chat history:', error);
  }
}

export async function closeDatabase() {
  if (pool) {
    await pool.end();
    console.log('MySQL connection pool closed');
  }
}
