import streamlit as st
import requests
import os
import tempfile
from io import BytesIO

# Backend URLs (Adjust to your backend URL if deployed)
FASTAPI_URL = "http://localhost:8000"  # Make sure to replace this with your FastAPI URL

# Language Options for translation
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

# Title of the app
st.title("Healthcare Translation Web App with Generative AI")

# Language Selection for translation
input_lang = st.selectbox("Select Input Language", list(LANGUAGES.keys()))
output_lang = st.selectbox("Select Output Language", list(LANGUAGES.keys()))

input_lang_code = LANGUAGES[input_lang]
output_lang_code = LANGUAGES[output_lang]

# Function to handle audio recording and uploading
def record_and_upload_audio():
    # Use Streamlit's file uploader to allow the user to upload an audio file
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "webm", "ogg"])

    if uploaded_audio is not None:
        # Show audio player for the uploaded file
        st.audio(uploaded_audio)

        # Process and upload the audio file when the button is pressed
        if st.button('Upload Audio'):
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                temp_audio_file.write(uploaded_audio.getvalue())
                temp_audio_file.close()

                # Use requests to send the file to the server
                with open(temp_audio_file.name, 'rb') as f:
                    files = {'file': (uploaded_audio.name, f, 'audio/webm')}
                    response = requests.post(f'{FASTAPI_URL}/upload_audio', files=files)

                if response.status_code == 200:
                    st.success("Audio uploaded successfully!")
                else:
                    st.error("Error uploading audio!")

                # Clean up the temporary file
                os.remove(temp_audio_file.name)

# Speech to Text Section (Transcription of uploaded audio)
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

            # Translate and generate audio when the "Translate and Speak" button is pressed
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
    # Record audio message
    record_and_upload_audio()

    st.header("Record Your Message")

    # Streamlit component for recording audio directly (no external HTML file required)
    st.write("Record your message directly below:")
    
    # Audio recording functionality: We will use pydub for this, but it requires manual testing.
    # Use Streamlit's file uploader to allow the user to upload an audio file.
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "webm", "ogg"])

    if uploaded_audio is not None:
        # Show audio player for the uploaded file
        st.audio(uploaded_audio)
        
        # Process and upload the audio file when the button is pressed
        if st.button('Upload Audio'):
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                temp_audio_file.write(uploaded_audio.getvalue())
                temp_audio_file.close()

                # Use requests to send the file to the server
                with open(temp_audio_file.name, 'rb') as f:
                    files = {'file': (uploaded_audio.name, f, 'audio/webm')}
                    response = requests.post(f'{FASTAPI_URL}/upload_audio', files=files)

                if response.status_code == 200:
                    st.success("Audio uploaded successfully!")
                else:
                    st.error("Error uploading audio!")

                # Clean up the temporary file
                os.remove(temp_audio_file.name)

# Main function to control app flow
if __name__ == "__main__":
    record_and_upload_audio()
