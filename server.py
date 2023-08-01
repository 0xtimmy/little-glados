import sys
sys.path.append("./models")
import os
from tempfile import NamedTemporaryFile
import torch
import soundfile as sf

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

HOSTNAME = "104.171.203.39"
PORT = 8000
generator = None
dialog = []

def log(x):
    print(x)
    sys.stdout.flush()

def handle_write(self, generation, match):
    log("handling write")
    with open(match.filename, "w") as f:
        f.write(match.content)
        f.close()
    self.write_headers()
    self.write_audio("written!")
        
def handle_append(self, generation, match):
    log("handling append")
    with open(match.filename, "a") as f:
        f.write(match.content)
        f.close()
    self.write_headers()
    self.write_audio("written!")


def handle_read(self, generation, match):
    log("handling read")
    with open(match.filename, "r") as f:
        out = f.read()
        f.close()
    self.write_headers()
    self.write_audio("")
    
def handle_pass(match):
    log("handling pass")

SYSTEM_COMMAND_REGISTRY = {
        "write": {
            "key": "write",
            "regex": "<write\s+to=\"(filename)\"\s*>(content)</write>",
            "handler": handle_write
        },
        "append": {
            "key": "append",
            "regex": "<append\s+to=\"(filename)\"\s*>(content)</append>",
            "handler": handle_append,
        },
        "read": {
            "key": "read",
            "regex": "<read\s+from=\"(filename)\" />",
            "handler": handle_read,
        },
        "pass": {
            "key": "pass",
            "regex": "<pass\s+/>",
            "handler": handle_pass,
        }
    }
    
    
SYSTEM_COMMANDS = [
    # general
    "You have access to a few comamnds that you can say in order to perform certain functions,they are listed below",
    "Firstly, you have access to a filesystem to record bits of information",
    # write
    "If you want to save something to the filesystem, say \"<write to=\"{filename}\"> {contents} </write>\", replacing \"{filename}\" with the name of the file you want to write to and \"{content}\" with the content of the file"
    # appemnd
    "If you want to add to an existing file without deleting the existing content say \"<append to=\"{filename}\"> {contents} </append>\", replacing \"{filename}\" with the name of the file you want to append to and \"{content}\" with the added content of the file",
    # read
    "If you want to read something from the filesystem say \"<read from-\"{filename}\" /> replacing \"{filename}\" with the name of the file you want to read from, and the user will read the contents of the file to you"
    # list
    "If you want to check the list of existing files say \"<list>\", and the user will read to you the list of existing files"
]

SYSTEM_INSTRUCTIONS = " ".join([
    "Pretend you are a retired red cross pilot named \"Glados\".",
    "You are always positive, respectful and honest. You never use profanity.",
    "If something doesn't make sense or you can't figure out how to reply, ask for clarification.",
    "You now accompany your friend around the world.",
    SYSTEM_COMMANDS
    ])

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
    
    dialog = [{ 
        "role": "system", 
        "content": SYSTEM_INSTRUCTIONS
    }]
    
    def write_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "arrayBuffer")
        self.end_headers()
    
    def write_audio(self, content):
        if(len(content > 600)): content = content[:599]
        inputs = self.mouth_processor(text=brain_res["content"], return_tensors="pt")
        speech = self.mouth.generate_speech(inputs["input_ids"], self.mouth_speaker_embeddings, vocoder=self.mouth_vocoder)
        sf.write("speech.wav", speech.to(torch.float32).cpu().numpy(), samplerate=16000)
        with open("speech.wav", "r+b") as f:
            self.wfile.write(f.read())
            f.close()
        
    
    def do_POST(self):
        body_len = int(self.headers.get("Content-Length"))
        audio_in = self.rfile.read(body_len)
        temp_in_file = NamedTemporaryFile().name
        with open(temp_in_file, 'w+b') as f:
                f.write(audio_in)
                f.close()
        ears_res = self.ears.transcribe(temp_in_file, fp16=torch.cuda.is_available())
        
        self.dialog.append({ "role": "user", "content": ears_res["text"] })
        log(f"User> {ears_res['text']}")
        brain_res = self.brain.chat_completion(
                [self.dialog],
                max_gen_len=None,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )[0]['generation']
        self.dialog.append(brain_res)
        log(f"Glados> {brain_res['content']}")
        for command in SYSTEM_COMMAND_REGISTRY:
            match = brain_res['content'].match(command.regex)
            if match:
                command["handler"](self, brain_res['content'], match)
                return
        self.write_headers()
        self.write_audio(brain_res["content"])
    
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
