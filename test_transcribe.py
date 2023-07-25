import torch
import sys
sys.path.insert(1, "./models")
import whisper
from tempfile import NamedTemporaryFile

listen = whisper.load_model("medium.en")

temp_file = NamedTemporaryFile().name
with open("speech.wav", "rb") as f:
    audio_in = f.read()
with open(temp_file, 'w+b') as f:
        f.write(audio_in)
        f.close()
text = listen.transcribe(temp_file, fp16=torch.cuda.is_available())
print(text)