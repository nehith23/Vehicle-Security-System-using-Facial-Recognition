# method to do offline text to speech using pyttsx3 library




import pyttsx3




def text_to_speech(user_text):

    try:
        engine = pyttsx3.init()
        engine.say(user_text)
        engine.runAndWait()
    except:
        print("ERROR: unknown error in text to speech")


if __name__ == "__main__":
    print(1)
    text_to_speech('Un-authorized person, access denied')
    print(2)
    text_to_speech('Welcome')

