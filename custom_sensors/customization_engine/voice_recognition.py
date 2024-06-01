from flask import Flask
import speech_recognition as sr
import whisper

model = whisper.load_model("base.en")
r = sr.Recognizer()

app = Flask(__name__)
@app.route("/")
def hello_world():

    with sr.Microphone(device_index=2) as source:
        print("Say something!")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source) 

    with open("recorded_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())
        print("Audio saved as recorded_audio.wav")


        result = model.transcribe("recorded_audio.wav")

        return result['text']


app.run(port=5001)