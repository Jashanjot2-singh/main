import requests
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from gtts import gTTS
from cryptography.fernet import Fernet
import tempfile
import logging
import os
import speech_recognition as sr

app = FastAPI()

# Configure Logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Encryption Key for Audio
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

# Language Mapping
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

OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace this with the actual URL for your Ollama server


def query_ollama_model(text: str, model: str = "llama3.2:1b"):
    """Function to query the Ollama model locally."""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [{"role": "user", "content": text}]
        }

        # Make the POST request to the Ollama server
        response = requests.post(OLLAMA_URL, json=data, headers=headers)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx errors

        # Get the response content (the translated text from Ollama)
        return response.json().get("choices", [])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Ollama model: {e}")
        return None


@app.post("/upload_audio")
async def upload_audio(file: UploadFile):
    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Check if the file exists and is readable
        if not os.path.exists(temp_file_path):
            return JSONResponse({"error": "Temporary file not found"}, status_code=400)

        with sr.AudioFile(temp_file_path) as source:
            audio = recognizer.record(source)  # Record the audio from file

        # Perform speech recognition
        try:
            text = recognizer.recognize_google(audio)  # Using Google's speech-to-text API
            logging.info(f"Recognized text: {text}")
        except sr.UnknownValueError:
            return JSONResponse({"error": "Could not understand the audio."}, status_code=400)
        except sr.RequestError as e:
            return JSONResponse({"error": f"Could not request results from Google Speech Recognition service; {e}"},
                                status_code=500)

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
    try:
        logging.info(f"Received text for translation: '{text}', from {input_lang_code} to {output_lang_code}")

        # Query the Ollama model for translation or processing
        translated_text = query_ollama_model(text, model="llama3.2:1b")

        if not translated_text:
            return JSONResponse({"error": "Failed to get translation from Ollama model."}, status_code=500)

        logging.info(f"Translated text: '{translated_text}'")

        # Generate Audio from the translated text
        tts = gTTS(translated_text, lang=output_lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)

            # Encrypt the file
            with open(temp_audio.name, "rb") as file:
                encrypted_data = cipher.encrypt(file.read())
            with open(temp_audio.name, "wb") as file:
                file.write(encrypted_data)

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
    try:
        # Decrypt and serve audio file
        decrypted_path = f"decrypted_{filename}"
        with open(filename, "rb") as file:
            encrypted_data = file.read()
        with open(decrypted_path, "wb") as file:
            file.write(cipher.decrypt(encrypted_data))

        return FileResponse(decrypted_path, media_type="audio/mp3")
    except Exception as e:
        logging.error(f"Error decrypting audio: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
