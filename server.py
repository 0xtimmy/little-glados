import http.server
import ssl

from llama import Llama

CKPT_DIR = "./parameters/llama-2-7b-chat"
TOKENIZER_PATH = "./models/llama/tokenizer.model"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
TEMPERATURE = 0.6
TOP_P = 0.9

HOSTNAME = "104.171.202.180"
PORT = 8000
generator = None
dialog = []

class GladosServer(http.server.BaseHTTPRequestHandler):
    
    test = 2
    generator = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=512,
            max_batch_size=1
    )
    dialog = []
    
    def do_POST(self):
        body_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(body_len).decode()

        self.dialog.append({ "role": "user", "content": body })
        response = self.generator.chat_completion(
                [self.dialog],
                max_gen_len=None,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )[0]['generation']
        self.dialog.append(response)
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes(response['content'], "utf-8"))
        self.wfile.write(bytes("hear you loud and clear", "utf-8"))

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
