import speech_recognition

r = speech_recognition.Recognizer()
with speech_recognition.AudioFile("./test.wav") as source:
    audio = r.record(source)
txt = r.recognize_google(audio, language="zh-tw")
print(txt)