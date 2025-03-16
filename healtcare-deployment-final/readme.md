# Healthcare Translation Web App

## Overview

The **Healthcare Translation Web App** is designed to assist users in translating speech from one language to another while also providing an audio version of the translated text. It is especially useful in healthcare settings where cross-language communication is crucial for patient care. The app performs the following tasks:

1. **Speech-to-Text**: Converts spoken language into text.
2. **Translation**: Translates the transcribed text from one language to another using the **Ollama Translation Model (LLaMA3.2:1b)**.
3. **Text-to-Speech**: Converts the translated text back into audio and securely encrypts it for delivery.
4. **Encryption and Decryption**: Ensures the security of audio files using encryption to prevent unauthorized access.

---

## Features

- **Speech Recognition**: Convert spoken input into text using Google Speech-to-Text.
- **Multi-language Support**: Translate between multiple languages.
- **Text-to-Speech**: Convert the translated text into audio using Google TTS.
- **Audio Encryption**: Secure the generated audio using encryption.
- **Simple Web Interface**: Built with **Streamlit**, the app allows users to upload audio files or record a message directly on the webpage.

---

## Libraries Used

1. **Streamlit**: For building the web interface and handling user interaction.
   - Installation: `pip install streamlit`
   
2. **pydub**: To process audio files (such as converting MP3 to WAV).
   - Installation: `pip install pydub`
   
3. **gTTS (Google Text-to-Speech)**: Converts text into speech.
   - Installation: `pip install gTTS`
   
4. **cryptography (Fernet)**: For encrypting and decrypting the audio files.
   - Installation: `pip install cryptography`
   
5. **SpeechRecognition**: For recognizing speech from audio files.
   - Installation: `pip install SpeechRecognition`
   
6. **requests**: To make HTTP requests to the Ollama API for translation.
   - Installation: `pip install requests`
   
7. **Ollama API (LLaMA 3.2:1b model)**: An advanced translation model used for translating text between languages.
   - LLaMA (Large Language Model by Ollama) is an AI model specialized in text generation and translation. The app utilizes the **LLaMA3.2:1b** model for translation purposes. This model helps translate between many languages and offers high-quality, efficient text-to-text translation.

---

## How it Works

### 1. **Speech-to-Text Function (`speech_to_text`)**

This function takes an audio file (WAV/MP3 format) as input and converts the speech to text using the **Google Speech Recognition** API. The recognized text is then returned for further processing.

```bash
def speech_to_text(audio_path: str):
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
```
2. Translation Function (query_ollama_model)
The translation function sends a request to the Ollama API to translate text. It uses the LLaMA3.2:1b model to translate the input text from one language to another.

```
def query_ollama_model(text: str, input_lang_code: str, output_lang_code: str, model: str = "llama3.2:1b"):
    """
    Function to query the Ollama model for translation.
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
```
3. Text-to-Speech Function (translate_and_speak)
Once the text is translated, this function uses Google Text-to-Speech (gTTS) to convert the translated text into audio. It saves the audio to a temporary file and encrypts it using the Fernet cipher.

```
def translate_and_speak(text: str, input_lang_code: str, output_lang_code: str):
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
```
4. Audio Encryption and Decryption Functions
To ensure the security of the generated audio file, encryption and decryption functions are used. The encrypted audio file is saved, and users can later decrypt it to listen to the translated message.
```
def decrypt_audio_file(encrypted_audio_file: str):
    decrypted_path = f"decrypted_{os.path.basename(encrypted_audio_file)}"
    with open(encrypted_audio_file, "rb") as file:
        encrypted_data = file.read()
    with open(decrypted_path, "wb") as file:
        file.write(cipher.decrypt(encrypted_data))

    return decrypted_path
```
User Interface (Streamlit)
The app uses Streamlit to provide a simple and interactive web interface. Here, users can:

Select the input and output languages for translation.
Upload an audio file or record a message.
The app will display the transcribed text, translated text, and provide an audio player for listening to the translated message.
python
Copy
# Streamlit interface for recording audio or uploading a file
```bash
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
            
            if st.button("Translate and Speak"):
                translated_text, audio_file_path = translate_and_speak(text, input_lang_code, output_lang_code)
                if translated_text and audio_file_path:
                    st.write(f"Translated Text: {translated_text}")
                    decrypted_audio_path = decrypt_audio_file(audio_file_path)
                    st.audio(decrypted_audio_path)
                else:
                    st.error("Error during translation or speech generation")
```
How to Run the App

Install the required dependencies:

```bash
pip install streamlit pydub gTTS cryptography SpeechRecognition requests
```
Set up the Ollama API:

You need to have a running instance of the Ollama API with the LLaMA3.2:1b model. The URL for the API is specified as http://localhost:11434/api/generate. Make sure the server is up and running.

Run the Streamlit app:

```bash
streamlit run app.py
```
Open the web interface in your browser, and start using the app to convert speech into translated audio messages.

Explanation of the LLaMA3.2:1b Model:

The LLaMA3.2:1b model, developed by Ollama, is a large language model (LLM) designed for tasks such as natural language processing (NLP), machine translation, and text generation. It is part of the LLaMA (Large Language Model Meta AI) family, known for its high efficiency and performance in text generation tasks.

The LLaMA3.2:1b model in this app is specifically used to:

Translate text from one language to another.
Provide high-quality translations across many languages, ensuring users can communicate effectively in a healthcare setting.
The Ollama API provides access to this model, allowing the app to send text and receive translations in real-time.

Conclusion

This Healthcare Translation Web App is a powerful tool for improving communication in healthcare environments, particularly when dealing with patients who speak different languages. By combining speech recognition, machine translation, and text-to-speech with secure audio encryption, this app provides an efficient and secure solution for overcoming language barriers.

License

This project is licensed under the MIT License - see the LICENSE file for details.
