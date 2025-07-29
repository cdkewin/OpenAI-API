import openai
import os
import pyttsx3
import tempfile
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

def chat_with_openai(messages):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wav.write(temp_file.name, fs, audio)
    print(f"Audio recorded to {temp_file.name}")
    return temp_file.name

def recognize_speech_whisper():
    audio_path = record_audio()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        return None
    openai.api_key = api_key
    with open(audio_path, "rb") as audio_file:
        try:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # Restrict transcription to English
            )
            print("You said:", transcript.text)
            return transcript.text
        except Exception as e:
            print(f"Whisper API error: {e}")
            return None

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def test_openai_api():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        return False
    openai.api_key = api_key
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print("OpenAI API test successful. Response:", response.choices[0].message.content)
        return True
    except Exception as e:
        print("OpenAI API test failed:", e)
        return False

if __name__ == "__main__":
    print("Testing OpenAI API connectivity...")
    if not test_openai_api():
        print("Exiting due to OpenAI API connectivity issues.")
        exit(1)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    try:
        while True:
            user_input = recognize_speech_whisper()
            if user_input:
                messages.append({"role": "user", "content": user_input})
                try:
                    reply = chat_with_openai(messages)
                    print("Assistant:", reply)
                    speak_text(reply)
                    messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    print("Error:", e)
            else:
                print("Please try speaking again.")
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
