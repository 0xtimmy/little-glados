import sys

sys.path.insert(1, "./models")

import io
import speech_recognition as sr

from time import time
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
import requests

from pydub import AudioSegment
from pydub.playback import play

ENERGY_THRESHOLD = 1000
RECORD_TIMEOUT = 1
PHRASE_TIMEOUT = 2
SAMPLE_RATE = 16000

HOSTNAME = "104.171.202.34"

def log(x):
    print(x)
    sys.stdout.flush()

def main():
    log("Setting up...")
    last_sample = bytes()
    data_queue = Queue()

    # Recorders Setup
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
    
    recorder.listen_in_background(source, record_callback, phrase_time_limit=RECORD_TIMEOUT)
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
                if time() - t_last_sample > PHRASE_TIMEOUT:
                    if phrase_started:
                        phrase_started = False
                        
                        log("--- Phrase Complete ---")
                        
                        with open(temp_file, "rb") as f:
                            res = requests.post(f"https://{HOSTNAME}:8000", data=f.read(), verify=False)
                            if(res.status_code == 200):
                                with open("speech.wav", "w+b") as f:
                                    f.write(res.content)
                                    f.close()
                                sound = AudioSegment.from_wav("speech.wav")
                                play(sound)
                            elif(res.status_code == 204):
                                log("--- Glados has decided not to say anything ---")
                            
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
