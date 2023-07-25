import sys
sys.path.append("./models")
from tempfile import NamedTemporaryFile
import torch

import http.server
import ssl

#models
from llama import Llama

import whisper

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

CKPT_DIR = "./parameters/llama-2-7b-chat"
TOKENIZER_PATH = "./models/llama/tokenizer.model"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
TEMPERATURE = 0.6
TOP_P = 0.9

HOSTNAME = "104.171.203.248"
PORT = 8000
generator = None
dialog = []

def log(x):
    print(x)
    sys.stdout.flush()

class GladosServer(http.server.BaseHTTPRequestHandler):
    
    test = 2
    ears = whisper.load_model("medium.en")
    brain = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=512,
            max_batch_size=1
    )
    
    mouth = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    mouth_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    mouth_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    mouth_speaker_embeddings = torch.tensor(load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")[7306]["xvector"]).unsqueeze(0)
    
    dialog = []
    
    def do_POST(self):
        body_len = int(self.headers.get("Content-Length"))
        audio_in = self.rfile.read(body_len)
        temp_file = NamedTemporaryFile().name
        with open(temp_file, 'w+b') as f:
                f.write(audio_in)
                f.close()
        ears_res = self.ears.transcribe(temp_file, fp16=torch.cuda.is_available())
        
        self.dialog.append({ "role": "user", "content": ears_res["text"] })
        log(f"User> {ears_res["text"]}")
        brain_res = self.brain.chat_completion(
                [self.dialog],
                max_gen_len=None,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )[0]['generation']
        self.dialog.append(brain_res)
        log(f"Glados> {brain_res["content"]}")
        if(len(brain_res["content"]) > 600): brain_res["content"] = brain_res["content"][:599]
        inputs = self.mouth_processor(text=brain_res["content"], return_tensors="pt")
        speech = self.mouth.generate_speech(inputs["input_ids"], self.mouth_speaker_embeddings, vocoder=self.mouth_vocoder)
        
        self.send_response(200)
        self.send_header("Content-type", "arrayBuffer")
        self.end_headers()
        self.wfile.write(speech.cpu().numpy().tobytes())
    
    #def do_POST(self):
    #    body_len = int(self.headers.get('Content-Length'))
    #    body = self.rfile.read(body_len).decode()
    #
    #    self.dialog.append({ "role": "user", "content": body })
    #    response = self.generator.chat_completion(
    #            [self.dialog],
    #            max_gen_len=None,
    #            temperature=TEMPERATURE,
    #            top_p=TOP_P,
    #        )[0]['generation']
    #    self.dialog.append(response)
    #    self.send_response(200)
    #    self.send_header("Content-type", "text/plain")
    #    self.end_headers()
    #    self.wfile.write(bytes(response['content'], "utf-8"))

# open server
def start():
    # open https server
    print("starting server")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain("cert.pem", None)
    server = http.server.HTTPServer((HOSTNAME, PORT), GladosServer)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    try:
        print("serving")
        server.serve_forever()
        print("server started")
    except KeyboardInterrupt:
        pass
    server.server_close()
    
if __name__ == '__main__':
    start()
