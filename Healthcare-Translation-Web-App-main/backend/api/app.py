import os
import tempfile
import logging
import requests
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from gtts import gTTS
from cryptography.fernet import Fernet
import speech_recognition as sr

# Initialize FastAPI application
app = FastAPI()

# Configure logging to write logs into 'app.log' file
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Generate an encryption key for securing audio files
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

# Language Mapping for supported languages
LANGUAGE_MAPPING = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Arabic": "ar",
    "Hindi": "hi",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Turkish": "tr"
}

# URL of the local Ollama model API server
OLLAMA_URL = "http://localhost:11434/api/generate"


def query_ollama_model(text: str, model: str = "llama3.2:1b"):
    """
    Function to query the Ollama model locally and get a translated response.

    Args:
        text (str): The input text to be processed or translated.
        model (str, optional): The model identifier to use for processing (default is "llama3.2:1b").

    Returns:
        str: The response text (translated or processed).
    """
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [{"role": "user", "content": text}]
        }

        # Send POST request to Ollama model API
        response = requests.post(OLLAMA_URL, json=data, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP error status codes

        # Extract and return the translated or processed text from the API response
        return response.json().get("choices", [])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Ollama model: {e}")
        return None


@app.post("/upload_audio")
async def upload_audio(file: UploadFile):
    """
    Endpoint to handle audio file uploads, recognize speech, and return recognized text.

    Args:
        file (UploadFile): The audio file uploaded by the user.

    Returns:
        JSONResponse: A JSON response containing the recognized text or an error message.
    """
    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Initialize speech recognizer
        recognizer = sr.Recognizer()

        # Check if the temporary file exists
        if not os.path.exists(temp_file_path):
            return JSONResponse({"error": "Temporary file not found"}, status_code=400)

        # Recognize speech from the uploaded audio
        with sr.AudioFile(temp_file_path) as source:
            audio = recognizer.record(source)

        # Perform speech-to-text recognition
        try:
            text = recognizer.recognize_google(audio)  # Using Google's speech-to-text API
            logging.info(f"Recognized text: {text}")
        except sr.UnknownValueError:
            return JSONResponse({"error": "Could not understand the audio."}, status_code=400)
        except sr.RequestError as e:
            return JSONResponse({"error": f"Google Speech Recognition request failed: {e}"}, status_code=500)

        # Return recognized text as a JSON response
        return JSONResponse({"text": text})

    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/translate/")
async def translate_and_speak(
        text: str = Form(...),
        input_lang_code: str = Form(...),
        output_lang_code: str = Form(...),
):
    """
    Endpoint to translate text and generate speech audio for the translated text.

    Args:
        text (str): The text to be translated.
        input_lang_code (str): The input language code (e.g., "en" for English).
        output_lang_code (str): The output language code (e.g., "es" for Spanish).

    Returns:
        JSONResponse: A JSON response containing the original and translated text and the path to the audio file.
    """
    try:
        logging.info(f"Received text for translation: '{text}', from {input_lang_code} to {output_lang_code}")

        # Query the Ollama model for translation or processing
        translated_text = query_ollama_model(text, model="llama3.2:1b")

        # Check if the translation was successful
        if not translated_text:
            return JSONResponse({"error": "Failed to get translation from Ollama model."}, status_code=500)

        logging.info(f"Translated text: '{translated_text}'")

        # Generate speech from the translated text using Google Text-to-Speech
        tts = gTTS(translated_text, lang=output_lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)

            # Encrypt the generated audio file
            with open(temp_audio.name, "rb") as file:
                encrypted_data = cipher.encrypt(file.read())
            with open(temp_audio.name, "wb") as file:
                file.write(encrypted_data)

        # Return JSON with the original text, translated text, and path to the encrypted audio file
        return JSONResponse({
            "original_text": text,
            "translated_text": translated_text,
            "audio_file": temp_audio.name
        })

    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return JSONResponse({"error": f"An error occurred during translation: {str(e)}"}, status_code=500)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Endpoint to serve the decrypted audio file.

    Args:
        filename (str): The name of the encrypted audio file.

    Returns:
        FileResponse: A response containing the decrypted audio file.
    """
    try:
        # Decrypt the audio file before serving it
        decrypted_path = f"decrypted_{filename}"
        with open(filename, "rb") as file:
            encrypted_data = file.read()

        # Write the decrypted audio data to a temporary file
        with open(decrypted_path, "wb") as file:
            file.write(cipher.decrypt(encrypted_data))

        # Return the decrypted audio file
        return FileResponse(decrypted_path, media_type="audio/mp3")

    except Exception as e:
        logging.error(f"Error decrypting audio: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
