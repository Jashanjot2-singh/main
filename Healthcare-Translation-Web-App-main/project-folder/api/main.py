from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from gtts import gTTS
from googletrans import Translator
from cryptography.fernet import Fernet
import tempfile
import logging
import os

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
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish"
}

@app.post("/translate/")
async def translate_and_speak(
    text: str = Form(...),
    input_lang_code: str = Form(...),
    output_lang_code: str = Form(...)
):
    try:
        # Translate Text
        translator = Translator()
        translated_text = translator.translate(text, src=input_lang_code, dest=output_lang_code).text

        # Generate Audio
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
        return JSONResponse({"error": str(e)}, status_code=500)

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
