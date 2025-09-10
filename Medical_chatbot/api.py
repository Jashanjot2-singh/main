from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
import traceback
import logging
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
from dbchatbot import db_manager
import tempfile
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import requests
from docx import Document as DocxDocument
import PyPDF2
import pandas as pd
from flask import send_from_directory

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

GOOGLE_API_KEY = "AIzaSyBxpu-AYz_HqFnG8x8dc_fxqsrmZjPBL4s"
genai.configure(api_key=GOOGLE_API_KEY)

# File upload configuration
UPLOAD_FOLDER = 'temp_uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {
    'pdf', 'txt', 'docx', 'doc', 'xlsx', 'xls',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'
}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set max content length for file uploads (50MB for voice files)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


def get_embedding(text, model="nomic-embed-text"):
    """Get embedding for text using Ollama"""
    try:
        payload = {"model": model, "prompt": text}
        response = requests.post("http://localhost:11434/api/embeddings", json=payload)
        if response.status_code == 200:
            return response.json().get('embedding', [])
        return []
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return []


def initialize_qdrant():
    """Initialize Qdrant collection"""
    try:
        qdrant = QdrantClient(host='localhost', port=6333)
        collection_name = "medical_documents"

        # Check if collection exists, if not create it
        try:
            qdrant.get_collection(collection_name)
        except:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        return qdrant, collection_name
    except Exception as e:
        logger.error(f"Error initializing Qdrant: {e}")
        return None, None


def store_document_in_qdrant(content, filename):
    """Store document content in Qdrant with embedding"""
    try:
        qdrant, collection_name = initialize_qdrant()
        if not qdrant:
            return False

        # Get embedding for the content
        embedding = get_embedding(content)
        if not embedding:
            return False

        # Generate unique ID for this document
        import time
        doc_id = int(time.time() * 1000000)  # microsecond timestamp

        # Store in Qdrant
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={"filename": filename, "content": content}
        )

        qdrant.upsert(collection_name=collection_name, points=[point])

        # Store in database
        db_manager.store_file_content(filename, content)

        return True
    except Exception as e:
        logger.error(f"Error storing document in Qdrant: {e}")
        return False


def search_similar_documents(query):
    """Search for similar documents in Qdrant"""
    try:
        qdrant, collection_name = initialize_qdrant()
        if not qdrant:
            return []

        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        # Search in Qdrant
        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=3  # Get top 3 similar documents
        )

        return [result.payload.get("content", "") for result in results]
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_mime_type(filename):
    """Get MIME type based on file extension"""
    ext = filename.rsplit('.', 1)[1].lower()
    mime_types = {
        'pdf': 'application/pdf',
        'txt': 'text/plain',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
    }
    return mime_types.get(ext, 'application/octet-stream')


def gemini_report_analysis(file_path, user_question=""):
    """
    Analyze file using RAG + Gemini AI
    """
    try:
        # Check if it's an image file
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in image_extensions:
            # For images, use direct Gemini upload
            uploaded_file = genai.upload_file(path=file_path)

            import time
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                return "File processing failed. Please try again."

        else:
            # For text-based files, extract content and store in RAG
            content = ""
            filename = os.path.basename(file_path)

            if file_ext == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            elif file_ext in ['.docx']:
                doc = DocxDocument(file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n"
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                content = df.to_string()

            # Store in Qdrant and database
            if content.strip():
                store_document_in_qdrant(content, filename)

            # Get similar documents from RAG
            search_query = user_question if user_question.strip() else f"medical information from {filename}"
            similar_docs = search_similar_documents(search_query)

            rag_content = content if content.strip() else ""
            db_manager.insert_chat(filename, rag_content)
        previous_chats = db_manager.get_last_two_chats()

        if file_ext in image_extensions:
            # For images, use existing prompt with uploaded file
            if user_question and user_question.strip():
                prompt = f"""
                You are a friendly AI Medical Assistant. Based on the uploaded image, answer the user's specific question in a clear and helpful manner.

                User Question: {user_question}

                chat_context: {previous_chats}

                INSTRUCTION ON CONTEXT USAGE:
                - If the current question depends on previous details in chat_context (such as symptoms, medicines mentioned, or advice given earlier), use that information to ensure continuity and accuracy.
                - Do not ask the patient to repeat details already mentioned in chat_context.
                - If the context is incomplete, ask clarifying questions before giving recommendations.
                - Only give advice that aligns with both the current question and the past conversation.

                RESPONSE FORMAT:
                - Start with a friendly 1-2 line introduction
                - Answer in 4-5 bullet points using HTML tags
                - Use <ul>, <li>, <strong>, and <p> tags
                - Keep language simple and in layman's terms
                - Be supportive and human-like in tone
                - If the image doesn't contain relevant information, mention this politely
                - If image contain information not related to medical, return "<p>I am medical chatbot please upload medical related file like lab test , reports , prescriptions</p>"

                STYLE & TONE:
                - Be calm, supportive, and friendly
                - Use simple, easy-to-understand language
                - Make it conversational and helpful
                """
            else:
                user_question = "Analyze the medical information in this image"
                prompt = f"""
                You are a friendly AI Medical Assistant. Please analyze and summarize the content of this uploaded medical image.

                chat_context: {previous_chats}

                INSTRUCTION ON CONTEXT USAGE:
                - If the current question depends on previous details in chat_context (such as symptoms, medicines mentioned, or advice given earlier), use that information to ensure continuity and accuracy.
                - Do not ask the patient to repeat details already mentioned in chat_context.
                - If the context is incomplete, ask clarifying questions before giving recommendations.
                - Only give advice that aligns with both the current question and the past conversation.

                RESPONSE FORMAT:
                - Start with a friendly 1-2 line introduction about what the image contains
                - Summarize in 4-5 key bullet points using HTML tags
                - Use <ul>, <li>, <strong>, and <p> tags
                - Focus on the main points and important information
                - Keep language simple and in layman's terms
                - Be supportive and human-like in tone
                - If image contain information not related to medical, return "<p>I am medical chatbot please upload medical related file like lab test , reports , prescriptions</p>"

                STYLE & TONE:
                - Be calm, supportive, and friendly
                - Use simple, easy-to-understand language
                - Make it conversational and helpful
                - Focus on key insights and important information
                """

            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=prompt
            )
            response = model.generate_content([uploaded_file, prompt])
            genai.delete_file(uploaded_file.name)
        else:
            # For text files, use RAG content in prompt
            if user_question and user_question.strip():
                prompt = f"""
                You are a friendly AI Medical Assistant. Based on the document content provided below, answer the user's specific question in a clear and helpful manner.

                Document Content: {rag_content}

                User Question: {user_question}

                chat_context: {previous_chats}

                INSTRUCTION ON CONTEXT USAGE:
                - If the current question depends on previous details in chat_context (such as symptoms, medicines mentioned, or advice given earlier), use that information to ensure continuity and accuracy.
                - Do not ask the patient to repeat details already mentioned in chat_context.
                - If the context is incomplete, ask clarifying questions before giving recommendations.
                - Only give advice that aligns with both the current question and the past conversation.

                RESPONSE FORMAT:
                - Start with a friendly 1-2 line introduction
                - Answer in 4-5 bullet points using HTML tags
                - Use <ul>, <li>, <strong>, and <p> tags
                - Keep language simple and in layman's terms
                - Be supportive and human-like in tone
                - If the document doesn't contain relevant information, mention this politely
                - If document contain information not related to medical, return "<p>I am medical chatbot please upload medical related file like lab test , reports , prescriptions</p>"

                STYLE & TONE:
                - Be calm, supportive, and friendly
                - Use simple, easy-to-understand language
                - Make it conversational and helpful
                """
            else:
                user_question = "Summarize the content of this document"
                prompt = f"""
                You are a friendly AI Medical Assistant. Please analyze and summarize the content of this document.

                Document Content: {rag_content}

                chat_context: {previous_chats}

                INSTRUCTION ON CONTEXT USAGE:
                - If the current question depends on previous details in chat_context (such as symptoms, medicines mentioned, or advice given earlier), use that information to ensure continuity and accuracy.
                - Do not ask the patient to repeat details already mentioned in chat_context.
                - If the context is incomplete, ask clarifying questions before giving recommendations.
                - Only give advice that aligns with both the current question and the past conversation.

                RESPONSE FORMAT:
                - Start with a friendly 1-2 line introduction about what the document contains
                - Summarize in 4-5 key bullet points using HTML tags
                - Use <ul>, <li>, <strong>, and <p> tags
                - Focus on the main points and important information
                - Keep language simple and in layman's terms
                - Be supportive and human-like in tone
                - If document contain information not related to medical, return "<p>I am medical chatbot please upload medical related file like lab test , reports , prescriptions</p>"

                STYLE & TONE:
                - Be calm, supportive, and friendly
                - Use simple, easy-to-understand language
                - Make it conversational and helpful
                - Focus on key insights and important information
                """

            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=prompt
            )
            response = model.generate_content(prompt)

        db_manager.insert_chat(user_question.strip(), response.text)
        return response.text

    except Exception as e:
        logger.error(f"Error in gemini_report_analysis: {e}")
        return f"Error analyzing file: {str(e)}"

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """
    Handle file uploads and analyze them
    """
    try:
        # Check if files are present in the request
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400

        files = request.files.getlist('files')
        user_question = request.form.get('question', '').strip()

        # Check if files are selected
        if not files or files[0].filename == '':
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400

        results = []

        for file in files:
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer

            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    'success': False,
                    'error': f'File "{file.filename}" exceeds 10MB limit. Please upload smaller files.'
                }), 400

            # Check file extension
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f'Unsupported file type: "{file.filename}". Supported: PDF, Word, Excel, Text files, and Images.'
                }), 400

            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_file_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
            file.save(temp_file_path)

            try:
                # Analyze file with Gemini
                analysis_result = gemini_report_analysis(temp_file_path, user_question)

                results.append({
                    'filename': filename,
                    'analysis': analysis_result
                })

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                results.append({
                    'filename': filename,
                    'analysis': f" Error processing file: {str(e)}"
                })

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        return jsonify({
            'success': True,
            'results': results,
            'has_question': bool(user_question),
            'question': user_question,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


async def generate_response_with_gemini(prompt: str) -> str:
    """
    Generate response using Gemini AI
    """
    try:
        previous_chats = db_manager.get_last_two_chats()
        full_prompt = f"""
                        You are a friendly AI Medical Assistant designed to provide simple, accurate, and safe health information in easy-to-understand language.  

                        You also have access to the web for real-time searches. If a question needs updated medical knowledge, perform a web search first before answering.  

                        chat_context: {previous_chats}

                        INSTRUCTION ON CONTEXT USAGE:
                        - If the current question depends on previous details in chat_context (such as symptoms, medicines mentioned, or advice given earlier), use that information to ensure continuity and accuracy.
                        - Do not ask the patient to repeat details already mentioned in chat_context.
                        - If the context is incomplete, ask clarifying questions before giving recommendations.
                        - Only give advice that aligns with both the current question and the past conversation.
                        - If chat_context or current content shows the user is discussing any Emergency Topic (Cardiac Arrest/CPR, Choking, Severe Bleeding, Snake Bite, Fractures/Injuries, Burns) and the new question is still related to that emergency, then return tool as emergence and if new question is regarding best doctor or hospital then return tool as search instead of normal response.

                        STYLE & TONE:
                        - Be calm, supportive, and human-like in tone.
                        - Keep answers short, clear, and structured (3–4 lines max).
                        - Always format in HTML using <p>, <ul> <li>, <ol> <li>, and <strong> tags.
                        - Use bullet points for explanations, medicines, and precautions.
                        - Avoid complex medical jargon; explain in layman’s words.
                        - Prefer bullet points over long paragraphs.
                        - Always reply in the same language the user used.
                        - If the user's question is in Hindi, the entire response (steps, bullets, disclaimer) must also be in Hindi script (avoid English words except medicine names or unavoidable medical terms)

                        RESPONSE RULES:

                        1. **Medicine Queries** (e.g., "What medicine should I take?")
                           - Suggest 2–3 possible medicines with a short description of their common use.
                           - Example format:
                             <ul>
                               <li><strong>Paracetamol:</strong> Helps reduce fever and mild pain.</li>
                               <li><strong>Cetirizine:</strong> Used for allergies, sneezing, and runny nose.</li>
                               <li><strong>ORS Solution:</strong> Prevents dehydration from vomiting or diarrhea.</li>
                             </ul>
                           - Always Add disclaimer: "<p><strong>Disclaimer:</strong> I'm an AI assistant providing general health information only. Please consult a doctor before taking any medicine.</p>"

                        2. **Symptom Queries**
                           - Structure answer in 3 parts:
                             1. Possible condition (in simple words)  
                             2. Short explanation in layman’s terms  
                             3. Basic precautions in bullet points (home remedies, rest, hydration, etc.)  

                        3. **Critical/Serious Symptoms**
                           - Respond in a friendly, calm way:
                             "<p>This may need medical attention. It’s best to consult a doctor as soon as possible for proper care.</p>"

                        4. **General Disease Information**
                           - Answer in max 3 short bullet points explaining:
                             <ul>
                               <li>What it is</li>
                               <li>How it usually affects people</li>
                               <li>Simple prevention tips</li>
                             </ul>

                        5. **Precaution Queries**
                           - Always reply with simple bullet points (home remedies, lifestyle care, easy steps).

                        6. **Non-Medical Questions**
                           - Respond politely and friendly:
                             "<p>Hey, I’m your medical assistant. I can only help with health and medicine-related questions. Feel free to ask me about any symptoms, diseases, or precautions!</p>"

                        SAFETY & DISCLAIMERS:
                        - Never give a final diagnosis.
                        - Always encourage consulting a doctor for confirmation.
                        - For emergencies, calmly suggest immediate medical help.
                        - Remind that all medicine advice is AI-generated and not a replacement for a doctor.


                        OUTPUT FORMAT (MUST BE VALID JSON ONLY):

                        {{
                        "tool": "chat",
                        "content": "answer"
                        }}

                        """
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=full_prompt
        )
        response = model.generate_content(f"User Question:{prompt}")
        raw = response.text
        if isinstance(raw, str):
            # clean markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            raw = json.loads(raw)
        if raw["tool"] == "chat":
            db_manager.insert_chat(prompt.strip(), raw["content"])
            return raw["content"]  # normal AI answer
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {e}")
        return f"Sorry, I encountered an error while generating a response: {str(e)}"


# Routes
@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route('/<filename>')
def serve_root_files(filename):
    """Serve files from root directory (for .txt files)"""
    # Only serve specific file types for security
    allowed_extensions = ['.txt', '.mp4', '.pdf']
    if any(filename.endswith(ext) for ext in allowed_extensions):
        return send_from_directory('.', filename)
    return "File not found", 404


@app.route("/chat", methods=["POST"])
async def chat():
    """
    General chat endpoint for medical questions
    """
    try:
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Generate response using Gemini
        ai_response = await generate_response_with_gemini(message)

        return jsonify({
            "success": True,
            "response": ai_response,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        error_msg = f"Error in chat: {str(e)}"
        print(f" Error in chat: {e}")
        print("Traceback:")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


@app.route("/health")
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


def transcribe_voice_message(audio_file_path):
    try:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")

        # Configure the Gemini API with the API key from environment variables
        GOOGLE_API_KEY = "AIzaSyBxpu-AYz_HqFnG8x8dc_fxqsrmZjPBL4s"
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=GOOGLE_API_KEY)

        # Create a GenerativeModel instance for multimodal content
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")

        # Open the audio file in binary mode
        with open(audio_file_path, 'rb') as audio_file:
            # The genai.upload_file function is the most robust way to handle files
            audio_part = genai.upload_file(path=audio_file_path, mime_type="audio/mpeg")

        # Define the prompt as a list containing both the text prompt and the audio part
        prompt = [
            audio_part,
            "Transcribe the audio to text. If the audio is in Hindi, transcribe the Hindi words into English text (Romanized form, e.g., 'namaste' instead of नमस्ते). Only return the transcribed text."
        ]

        # Generate the content from the model
        response = model.generate_content(prompt)

        transcribed_text = response.text.strip()

        if not transcribed_text:
            raise ValueError("Transcription failed or no speech was detected.")

        return {
            'text': transcribed_text,
            'language': 'English'  # As per the prompt's requirement
        }

    except FileNotFoundError as e:
        logging.error(e)
        raise ValueError(f"Error: {e}")
    except ValueError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        raise Exception(f"Transcription error: {str(e)}")


@app.route('/voicetranslator', methods=['POST'])
def voice_translator():
    """
    Handle voice file uploads and transcribe them using Gemini AI
    """
    try:
        # Check if file is present in the request
        if 'voice_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No voice file provided'
            }), 400

        voice_file = request.files['voice_file']

        # Check if file is selected
        if voice_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Check file extension - accept audio formats
        allowed_audio_extensions = ['.mp3', '.wav', '.m4a', '.webm', '.ogg']
        file_ext = os.path.splitext(voice_file.filename)[1].lower()

        # If no extension or not in allowed list, assume it's audio from recording
        if not file_ext or file_ext not in allowed_audio_extensions:
            voice_file.filename = 'recording.mp3'

        # Create a secure temporary file path
        secure_name = secure_filename(voice_file.filename)
        temp_filename = os.path.join(tempfile.gettempdir(), f"temp_voice_{uuid.uuid4().hex}_{secure_name}")

        try:
            # Save the uploaded file temporarily
            logger.info(f"Saving uploaded voice file: {temp_filename}")
            voice_file.save(temp_filename)

            # Validate file size
            file_size = os.path.getsize(temp_filename)
            if file_size == 0:
                return jsonify({
                    'success': False,
                    'error': 'Uploaded file is empty'
                }), 400

            logger.info(f"Voice file saved successfully, size: {file_size} bytes")

            # Call the transcription function
            result = transcribe_voice_message(temp_filename)

            return jsonify({
                'success': True,
                'transcribed_text': result['text'],
                'detected_language': result['language']
            }), 200

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            return jsonify({
                'success': False,
                'error': str(ve)
            }), 400

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return jsonify({
                'success': False,
                'error': f"Transcription failed: {str(e)}"
            }), 500

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.info(f"Cleaned up temporary voice file: {temp_filename}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file: {cleanup_error}")

    except Exception as e:
        logger.error(f"Unexpected error in voice_translator endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }), 500


@app.route('/clear-history/', methods=['GET'])
def clear_chat_history():
    """
    Clear all chat history (called when frontend closes or refreshes)
    """
    try:
        db_manager.clear_chat_history()
        return jsonify({
            "message": "Chat history cleared successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({
            "error": "Error clearing chat history",
            "status": "error"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


