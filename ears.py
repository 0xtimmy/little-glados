import sys

sys.path.insert(1, "./models")

import io
import speech_recognition as sr
import whisper
import torch

from time import time
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
import requests

ENERGY_THRESHOLD = 1000
RECORD_TIMEOUT = 2
PHRASE_TIMEOUT = 1000
SAMPLE_RATE = 16000

def main():
    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()
    
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    recorder.dynamic_energy_threshold = False
    
    source = sr.Microphone(sample_rate=SAMPLE_RATE)
    
    listen = whisper.load_model("medium.en")
    
    record_timeout = RECORD_TIMEOUT
    phrase_timeout = PHRASE_TIMEOUT
    
    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)
        
    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)
    
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    
    print("Model loaded.\n")
    
    while True:
        try:
            now = time()

            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > phrase_timeout:
                    last_sample = bytes()
                    phrase_complete = True
                    print("> Phrase complete")
                phrase_time = now
                
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                    
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                    
                result = listen.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text']
                
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                    
                if phrase_complete:
                    # get response
                    print(f"> {transcription[-1]}")
                    transcription = transcription
                    
                sleep(0.25)
                    
        except KeyboardInterrupt:
            break
    
    print("\n\nTranscription:")
    for line in transcription:
        print(line)
    
if __name__ == '__main__':
    main()