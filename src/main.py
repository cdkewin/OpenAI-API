import openai
import os
import speech_recognition as sr
import pyttsx3

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

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}\nPlease check your internet connection or try again later.")
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
    while True:
        user_input = recognize_speech()
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
