import streamlit as st
import requests
import os
import tempfile
import time

# Backend URLs
FASTAPI_URL = "http://localhost:8000"  # Adjust to your backend URL if deployed

# Language options mapping: the key is the language name, and the value is the language code.
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

# Set the title of the web app
st.title("Healthcare Translation Web App with Generative AI")

# Language Selection - Let the user select the input and output languages
input_lang = st.selectbox("Select Input Language", list(LANGUAGES.keys()))
output_lang = st.selectbox("Select Output Language", list(LANGUAGES.keys()))

# Get the corresponding language codes from the selected languages
input_lang_code = LANGUAGES[input_lang]
output_lang_code = LANGUAGES[output_lang]

# Section Header for Voice to Text functionality
st.header("Voice to Text")

# Option for user to choose between uploading an audio file or recording a message
audio_option = st.radio("Choose Audio Input", ["Upload an Audio File", "Record a Message"])

# If the user chooses "Upload an Audio File"
if audio_option == "Upload an Audio File":
    # File uploader widget for audio file input (MP3 or WAV formats)
    audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

    # If an audio file is uploaded
    if audio_file:
        # Create a temporary file to store the uploaded audio locally
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_file.getvalue())  # Write uploaded file contents to temporary file
            temp_audio_file_path = temp_audio_file.name  # Store the temporary file path

        # Send the uploaded audio to the FastAPI backend for speech-to-text conversion
        files = {"file": open(temp_audio_file_path, "rb")}
        response = requests.post(f"{FASTAPI_URL}/upload_audio", files=files)

        # If the request is successful (status code 200)
        if response.status_code == 200:
            # Get the transcribed text from the API response
            text = response.json().get("text")
            st.write(f"Transcribed Text: {text}")

            # Button to trigger translation and text-to-speech generation
            if st.button("Translate and Speak"):
                # Send a request to translate the transcribed text
                translation_response = requests.post(f"{FASTAPI_URL}/translate/", data={
                    "text": text,
                    "input_lang_code": input_lang_code,
                    "output_lang_code": output_lang_code,
                })

                # If translation is successful
                if translation_response.status_code == 200:
                    # Get the translated text and audio file path from the response
                    translated_data = translation_response.json()
                    translated_text = translated_data.get("translated_text")
                    st.write(f"Translated Text: {translated_text}")

                    # Generate the URL for the translated audio file
                    audio_url = f"{FASTAPI_URL}/audio/{os.path.basename(translated_data['audio_file'])}"
                    # Provide an audio player for the translated speech
                    st.audio(audio_url)

        else:
            # If speech-to-text conversion fails, show an error
            st.error("Error during speech-to-text")

# If the user chooses "Record a Message"
elif audio_option == "Record a Message":
    # Display a header indicating message recording
    st.header("Record Your Message")

    # Embed an HTML audio recording interface for the user to record their voice message
    # We assume there is an HTML file (record_audio.html) to handle audio recording
    st.components.v1.html(open("record_audio.html", "r").read(), height=300)

    # Display a placeholder message to instruct the user during the recording
    st.write("Hold the 'Start Recording' button to record a message")
