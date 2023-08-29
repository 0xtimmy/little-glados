from datasets import load_dataset

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

mouth = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
mouth_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
mouth_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
mouth_speaker_embeddings = torch.tensor(load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")[3415]["xvector"]).unsqueeze(0)

speaker_embeddings = embeddings_dataset[3415]
print(speaker_embeddings)
