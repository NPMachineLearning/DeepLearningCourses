import os

import speech_recognition

print(speech_recognition.Microphone.list_microphone_names())
mic = speech_recognition.Microphone(device_index=1)
r = speech_recognition.Recognizer()
# with mic as source:
#     r.adjust_for_ambient_noise(source)
#     audio = r.listen(source)
# txt = r.recognize_google(audio, language="zh-tw")
# print(txt)

while True:
    print("Start speaking...", end="")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        txt = r.recognize_google(audio, language="zh-tw")
        print()
        print(txt)
        if "小算盤" in txt: os.system("calc")
        if txt == "離開": break
    except:
        print("\r", end="")
