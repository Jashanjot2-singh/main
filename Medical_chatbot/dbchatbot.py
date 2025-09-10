import mysql.connector
from mysql.connector import Error
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect_to_database()
        self.create_database_and_table()

    def connect_to_database(self):
        """Connect to MySQL server"""
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                user='root',  # Default XAMPP MySQL username
                password='',  # Default XAMPP MySQL password (empty)
                port=3306
            )
            logger.info("Connected to MySQL server successfully")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise e

    def create_database_and_table(self):
        """Create database and table if they don't exist"""
        try:
            cursor = self.connection.cursor()

            # Create database if not exists
            cursor.execute("CREATE DATABASE IF NOT EXISTS Sonu_chatbot")
            cursor.execute("USE Sonu_chatbot")

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    question TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.connection.commit()
            logger.info("Database and table created/verified successfully")

        except Error as e:
            logger.error(f"Error creating database/table: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()

    def insert_chat(self, question, response):
        """Insert chat data into database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE Sonu_chatbot")

            query = "INSERT INTO chat_history (question, response) VALUES (%s, %s)"
            cursor.execute(query, (question, response))

            self.connection.commit()
            logger.info("Chat data inserted successfully")

        except Error as e:
            logger.error(f"Error inserting chat data: {e}")
        finally:
            if cursor:
                cursor.close()

    def get_last_two_chats(self):
        """Get the last two chat entries"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE Sonu_chatbot")

            query = """
                SELECT question, response FROM chat_history 
                ORDER BY created_at DESC 
                LIMIT 2
            """
            cursor.execute(query)
            results = cursor.fetchall()

            # Reverse to get chronological order (oldest first)
            return list(reversed(results))

        except Error as e:
            logger.error(f"Error fetching chat history: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def clear_chat_history(self):
        """Clear all chat history"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE Sonu_chatbot")

            cursor.execute("DELETE FROM chat_history")
            self.connection.commit()
            logger.info("Chat history cleared successfully")

        except Error as e:
            logger.error(f"Error clearing chat history: {e}")
        finally:
            if cursor:
                cursor.close()

    def close_connection(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")


# Global database manager instance
db_manager = DatabaseManager()