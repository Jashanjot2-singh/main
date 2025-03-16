# Healthcare Translation Web App

🌟 **Real-time multilingual translation for patients and healthcare providers!**

This project is designed to bridge the language gap between patients and healthcare professionals by offering **speech-to-text translation** in real time. It leverages state-of-the-art **Generative AI** technologies to ensure seamless communication across multiple languages while ensuring **data security** and **user-friendliness**. 

---

## 📋 **Features**

### 🎙️ **Speech-to-Text**
- Captures speech input via the microphone.
- Utilizes Google Speech Recognition for accurate transcription in the selected language.

### 🌐 **Real-time Translation**
- Translates text between 14 languages using Google Translate.
- Handles diverse linguistic needs with precision.

### 🔊 **Text-to-Speech with Encryption**
- Converts the translated text to audio output using gTTS (Google Text-to-Speech).
- Encrypts audio files for secure storage and playback.

### 🔒 **Data Security**
- Uses the **Fernet encryption** protocol to secure audio files.
- Automatically decrypts files for playback and ensures encrypted files are removed post-use.

### 🖥️ **Streamlit-based Interface**
- Intuitive web app with customizable language settings.
- Mobile-compatible for on-the-go usage.

---

## 🛠️ **Technical Details**

### **Technology Stack**
- **Frontend**: Streamlit for creating the user interface.
- **Backend**: Python for core logic.
- **APIs**:
  - Google Speech Recognition for transcription.
  - Google Translate for translation.
  - gTTS for text-to-speech.
- **Security**:
  - Fernet encryption (part of the cryptography library) for securing audio files.
---
## 🚀 **User Guide**

1. **Select Languages**:
   - Choose the input and output languages from the sidebar.
2. **Start Speaking**:
   - Click the "Start Speaking" button and begin speaking into your microphone.
3. **Translation**:
   - View the original and translated text on the screen.
4. **Audio Output**:
   - Listen to the translated audio, securely processed for privacy.
---

## 🛡️ **Security and Privacy**
- **Encryption**: All audio files are encrypted using Fernet for enhanced security.
- **Data Removal**: Temporary files are deleted immediately after use to ensure privacy.

---

🚀 *Let’s break language barriers in healthcare, one word at a time!* 🌍
