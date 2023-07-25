import sys

sys.path.insert(1, "./models")

import io
import speech_recognition as sr
import whisper
import torch
import numpy

from time import time
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
import requests

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

ENERGY_THRESHOLD = 1000
RECORD_TIMEOUT = 1
PHRASE_TIMEOUT = 3
SAMPLE_RATE = 16000

BRAIN_IP = "104.171.202.53"

def log(x):
    print(x)
    sys.stdout.flush()

def main():
    log("Setting up...")
    last_sample = bytes()
    data_queue = Queue()
    
    # Whisper Setup
    listen = whisper.load_model("medium.en")
    
    # T5 Setup
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Recorders Setup
    record_timeout = RECORD_TIMEOUT
    phrase_timeout = PHRASE_TIMEOUT
    
    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    recorder.dynamic_energy_threshold = False
    
    source = sr.Microphone(sample_rate=SAMPLE_RATE)
    with source:
        recorder.adjust_for_ambient_noise(source)
        
    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)
    
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    log("Setup complete! Listening...")
    t_last_sample = time()
    phrase_started = False
    
    while True:
        try:

            if not data_queue.empty():
                phrase_started = True
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                    
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                    f.close()
                    
                t_last_sample = time()
            else:
                if time() - t_last_sample > 5:
                    if phrase_started:
                        phrase_started = False
                        
                        log("--- Phrase Complete ---")
                        
                        # Ears
                        #result = listen.transcribe(temp_file, fp16=torch.cuda.is_available())
                        #text = result['text']
                        #transcription.append(text)
                        #log(f"User> {transcription[-1]}")
                        
                        # Brain
                        #res = requests.post(f"https://{BRAIN_IP}:8000", data=transcription[-1], verify=False)
                        with open(temp_file, "rb") as f:
                            res = requests.pos(f"https://{BRAIN_IP}:8000", data=f.read(), verify=False)
                            sound = AudioSegment(res.content, sample_width=2, framerate=16000, channels=1)
                            play(sound)
                        #log(f"Glados> {res.text}")
                        
                        # Mouth
                        #inputs = processor(text=res.text, return_tensors="pt")
                        #speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                        #sf.write("speech.wav", speech.numpy(), samplerate=16000)
                        #sound = AudioSegment.from_wav("speech.wav")
                        #play(sound)

                        log("--- Waiting for input ---")
                        last_sample = bytes()
                        t_last_sample = time()
                    else:
                        t_last_sample = time()
                
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)
    
if __name__ == '__main__':
    main()