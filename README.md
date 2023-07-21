# Little Glados

## Architecture
- speech input: Whisper (local)
- brain: llama (remote)
- speech output: ??? (local)

## To run brain:
```
torchrun --nproc_per_node 1 chat.py --ckpt_dir parameters/llama-2-7b-chat --tokenizer_path llama/tokenizer.model
```
