import http.server
import ssl

import brain

HOSTNAME = "104.171.202.180"
PORT = 8000
generator = None
dialog = []

class GladosServer(http.server.BaseHTTPRequestHandler):
    
    test = 2
    generator = brain.boot()
    dialog = []
    
    def do_POST(self):
        body_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(body_len).decode()

        self.dialog.append({ "role": "user", "content": body })
        response = brain.respond(self.generator, self.dialog)
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
    

# handle input
def handle_promot(input: str):
    dialog.append([{ "role": "user", "content": input}])
    results = brain.respond(generator, dialog)
    response = results[0]['generation']['content']
    #dialog.append(["role": "glados", "content": response]) ???
    return response

# terminate model
def kill():
    # close https server
    generator = None
    dialog = []
    
if __name__ == '__main__':
    start()
