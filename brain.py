import sys
sys.path.insert(1, "./models")

from typing import Optional
from llama import Llama

CKPT_DIR = "./parameters/llama-2-7b-chat"
TOKENIZER_PATH = "./models/llama/tokenizer.model"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
TEMPERATURE = 0.6
TOP_P = 0.9

def boot():
    return "booted"
    #return Llama.build(
    #        ckpt_dir=CKPT_DIR,
    #        tokenizer_path=TOKENIZER_PATH,
    #        max_seq_len=512,
    #        max_batch_size=1
    #)

def respond(generator, dialog):
    return generator + dialog
    #return generator.chat_completion(
    #            dialog,
    #            max_gen_len=None,
    #            temperature=TEMPERATURE,
    #            top_p=TOP_P,
    #        )

def main():
    

    generator = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=512,
            max_batch_size=1
    )
    try:
        while True:
            user_input = input("> User: ")

            results = generator.chat_completion(
                [[{ "role": "user", "content": user_input }]],
                max_gen_len=None,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            #print("---")
            #print(results)
            #print("---")

            print(
                f"> Little Glados: {results[0]['generation']['content']}"
            )


    except KeyboardInterrupt:
        print("--- Exiting chat ---")

if __name__ == "__main__":
    main()
