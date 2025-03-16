import streamlit as st
import tempfile
import os
import wave
import io
#from pydub import AudioSegment
from gtts import gTTS
from cryptography.fernet import Fernet
import logging
import speech_recognition as sr
import requests

# Configure Logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Encryption Key for Audio
encryption_key = Fernet.generate_key()  # Generate a unique encryption key
cipher = Fernet(encryption_key)  # Initialize the cipher for encryption and decryption

# Language Mapping (for translation purposes)
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

# Ollama API URL for querying the translation model
llama_api_url = "http://localhost:11434/api/generate"  # URL for Ollama server, change accordingly


def query_ollama_model(text: str, input_lang_code: str, output_lang_code: str, model: str = "llama3.2:1b"):
    """
    Function to query the Ollama model for translation.

    Args:
    - text (str): The text to translate.
    - input_lang_code (str): Language code for the source language.
    - output_lang_code (str): Language code for the target language.
    - model (str): The model to use for translation (default is "llama3.2:1b").

    Returns:
    - str: The translated text.
    """
    # Format the prompt for translation request
    formatted_prompt = f"Translate this text from {LANGUAGE_MAPPING[input_lang_code]} to {LANGUAGE_MAPPING[output_lang_code]}: {text}"
    payload = {"model": model, "prompt": formatted_prompt, "stream": False}

    try:
        # Send the translation request to the Ollama API
        response = requests.post(llama_api_url, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
        return response.json().get('response', 'No response from LLaMA API')
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Ollama model: {e}")
        return None


def speech_to_text(audio_path: str):
    """
    Converts audio file to text using the speech_recognition library.

    Args:
    - audio_path (str): Path to the audio file to transcribe.

    Returns:
    - str: The transcribed text.
    """
    recognizer = sr.Recognizer()

    # Check if the file exists and is readable
    if not os.path.exists(audio_path):
        return None

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # Record the audio from file

    try:
        # Use Google's speech recognition API to transcribe the audio
        text = recognizer.recognize_google(audio)
        logging.info(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        logging.error("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None


def translate_and_speak(text: str, input_lang_code: str, output_lang_code: str):
    """
    Translates text using the Ollama model and generates speech from translated text.

    Args:
    - text (str): The text to translate and convert to speech.
    - input_lang_code (str): The language code of the input text.
    - output_lang_code (str): The language code for the output translation.

    Returns:
    - tuple: A tuple containing the translated text and the path to the audio file.
    """
    try:
        logging.info(f"Received text for translation: '{text}', from {input_lang_code} to {output_lang_code}")

        # Query the Ollama model for translation
        translated_text = query_ollama_model(text, input_lang_code, output_lang_code)

        if not translated_text:
            logging.error("Failed to get translation.")
            return None, None

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

        return translated_text, temp_audio.name

    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return None, None


def decrypt_audio_file(encrypted_audio_file: str):
    """
    Decrypts the encrypted audio file.

    Args:
    - encrypted_audio_file (str): Path to the encrypted audio file.

    Returns:
    - str: Path to the decrypted audio file.
    """
    decrypted_path = f"decrypted_{os.path.basename(encrypted_audio_file)}"
    with open(encrypted_audio_file, "rb") as file:
        encrypted_data = file.read()
    with open(decrypted_path, "wb") as file:
        file.write(cipher.decrypt(encrypted_data))

    return decrypted_path


def save_audio(audio_data, file_name="recorded_audio.wav"):
    """
    Saves the audio data to a file.

    Args:
    - audio_data (bytes): The raw audio data.
    - file_name (str): The name of the file to save the audio data to.

    Returns:
    - str: The path to the saved audio file.
    """
    with open(file_name, 'wb') as f:
        f.write(audio_data)
    return file_name


# HTML & JS for audio recording (embedded in the Streamlit app)
recording_html = """
<html>
  <body>
    <h3>Record your message</h3>
    <script>
      var recordButton = document.createElement("button");
      recordButton.innerHTML = "Start Recording";
      document.body.appendChild(recordButton);

      var stopButton = document.createElement("button");
      stopButton.innerHTML = "Stop Recording";
      document.body.appendChild(stopButton);
      stopButton.disabled = true;

      var audioChunks = [];
      var recorder;

      // Request for microphone access and start recording
      recordButton.onclick = function() {
        navigator.mediaDevices.getUserMedia({ audio: true })
          .then(function(stream) {
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = function(event) {
              audioChunks.push(event.data);
            };
            recorder.onstop = function() {
              var audioBlob = new Blob(audioChunks, { type: "audio/wav" });
              var audioUrl = URL.createObjectURL(audioBlob);
              var audio = new Audio(audioUrl);
              document.body.appendChild(audio);
              var downloadLink = document.createElement("a");
              downloadLink.href = audioUrl;
              downloadLink.download = "recorded_audio.wav";
              downloadLink.innerHTML = "Download Recorded Audio";
              document.body.appendChild(downloadLink);
            };
            recorder.start();
            stopButton.disabled = false;
            recordButton.disabled = true;
          })
          .catch(function(error) {
            alert("Unable to access microphone");
          });
      };

      // Stop recording and finish the process
      stopButton.onclick = function() {
        recorder.stop();
        stopButton.disabled = true;
        recordButton.disabled = false;
      };
    </script>
  </body>
</html>
"""

# Streamlit App
st.title("Healthcare Translation Web App with Generative AI")

# Language Selection
input_lang = st.selectbox("Select Input Language", list(LANGUAGE_MAPPING.keys()))
output_lang = st.selectbox("Select Output Language", list(LANGUAGE_MAPPING.keys()))

input_lang_code = input_lang  # Use the code directly
output_lang_code = output_lang  # Use the code directly

# Speech to Text Section
st.header("Voice to Text")

# Option for File Upload or Recording
audio_option = st.radio("Choose Audio Input", ["Upload an Audio File", "Record a Message"])

if audio_option == "Upload an Audio File":
    audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

    if audio_file:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_file.getvalue())
            temp_audio_file_path = temp_audio_file.name

        # Perform speech-to-text
        text = speech_to_text(temp_audio_file_path)
        if text:
            st.write(f"Transcribed Text: {text}")

            # Translate and generate audio
            if st.button("Translate and Speak"):
                translated_text, audio_file_path = translate_and_speak(text, input_lang_code, output_lang_code)
                if translated_text and audio_file_path:
                    st.write(f"Translated Text: {translated_text}")

                    # Audio File for Playback
                    decrypted_audio_path = decrypt_audio_file(audio_file_path)
                    st.audio(decrypted_audio_path)
                else:
                    st.error("Error during translation or speech generation")

        else:
            st.error("Error during speech-to-text")

elif audio_option == "Record a Message":
    st.header("Record Your Message")

    # Embed the HTML audio recording interface
    st.components.v1.html(recording_html, height=400)

    st.write("After recording your message, download the file and upload it here.")

# File upload section (for when the user records and downloads the audio file)
uploaded_file = st.file_uploader("Upload your recorded audio here (if saved locally)", type=["wav", "mp3"])

if uploaded_file:
    # Save the uploaded audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        audio_path = temp_file.name
        st.write(f"File uploaded successfully: {audio_path}")

        # Process the uploaded audio file (for example, play the audio)
        audio_file = AudioSegment.from_wav(audio_path)  # or .from_mp3() depending on the file type
        st.audio(audio_path)

        # Perform speech-to-text on the uploaded file
        text = speech_to_text(audio_path)
        if text:
            st.write(f"Transcribed Text: {text}")
            if st.button("Translate and Speak"):
                translated_text, audio_file_path = translate_and_speak(text, input_lang_code, output_lang_code)
                if translated_text and audio_file_path:
                    st.write(f"Translated Text: {translated_text}")
                    decrypted_audio_path = decrypt_audio_file(audio_file_path)
                    st.audio(decrypted_audio_path)
                else:
                    st.error("Error during translation or speech generation")
