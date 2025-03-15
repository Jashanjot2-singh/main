import streamlit as st
import requests
import os
import base64
from io import BytesIO
import time

# Backend URLs
FASTAPI_URL = "http://localhost:8000"  # Adjust to your backend URL if deployed

# Language Options
LANGUAGES = {
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

# Title
st.title("Healthcare Translation Web App with Generative AI")

# Language Selection
input_lang = st.selectbox("Select Input Language", list(LANGUAGES.keys()))
output_lang = st.selectbox("Select Output Language", list(LANGUAGES.keys()))

input_lang_code = LANGUAGES[input_lang]
output_lang_code = LANGUAGES[output_lang]

# Speech to Text Section
st.header("Voice to Text")

# Option for File Upload or Recording
audio_option = st.radio("Choose Audio Input", ["Upload an Audio File", "Record a Message"])

if audio_option == "Upload an Audio File":
    audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

    if audio_file:
        # Upload to FastAPI for speech-to-text
        files = {"file": audio_file.getvalue()}
        response = requests.post(f"{FASTAPI_URL}/speech-to-text/", files=files)
        if response.status_code == 200:
            text = response.json().get("text")
            st.write(f"Transcribed Text: {text}")

            # Translate and generate audio
            if st.button("Translate and Speak"):
                translation_response = requests.post(f"{FASTAPI_URL}/translate/", data={
                    "text": text,
                    "input_lang_code": input_lang_code,
                    "output_lang_code": output_lang_code,
                })
                if translation_response.status_code == 200:
                    translated_data = translation_response.json()
                    translated_text = translated_data.get("translated_text")
                    st.write(f"Translated Text: {translated_text}")

                    # Audio File for Playback
                    audio_url = f"{FASTAPI_URL}/audio/{os.path.basename(translated_data['audio_file'])}"
                    st.audio(audio_url)

        else:
            st.error("Error during speech-to-text")
else:
    # Title and description
    st.title("Record Your Message")
    st.write("Click the button below to start recording your message.")
    
    # Use Streamlit's file uploader to allow the user to upload an audio file
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "webm", "ogg"])

    if uploaded_audio is not None:
        # Show audio player
        st.audio(uploaded_audio)
        
        # Handle the audio file, process and upload it
        if st.button('Upload Audio'):
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                temp_audio_file.write(uploaded_audio.getvalue())
                temp_audio_file.close()
                
                # Use requests to send the file to the server
                with open(temp_audio_file.name, 'rb') as f:
                    files = {'file': (uploaded_audio.name, f, 'audio/webm')}
                    response = requests.post('/upload_audio', files=files)

                if response.status_code == 200:
                    st.success("Audio uploaded successfully!")
                else:
                    st.error("Error uploading audio!")

                # Clean up the temporary file
                os.remove(temp_audio_file.name)
