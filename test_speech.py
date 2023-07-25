from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from pydub import AudioSegment

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

inputs = processor(text="The quick brown fox jumped over the lazy dog", return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("speech.wav", speech.numpy(), samplerate=16000)
sound = AudioSegment.from_wav("speech.wav")
print(f"frame rate: {sound.frame_rate}")
print(f"sample width: {sound.sample_width}")
print(f"channels: {sound.channels}")
