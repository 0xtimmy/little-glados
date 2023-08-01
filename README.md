# Little Glados

World ending AGI, but small. Little Glados is a helpful robot designed to be spoken to, and run on standalone hardware such that it can be brought into your offline life! Whether that's being a smarter siri, riding around on your roomba like a cat, or carried in your backpack to cheat in school, little glados will be there to help! Little Glados will NOT try to kill you.

This protoype is built by chaining together speech->text, llm, and text->speech models (OpenAI's Whisper, Facebook's Llama, and Microsofts SpeechT5 respectively) to work as an AI you can talk to. It is right now setup for simple chat (like ChatGPT) that uses a local mic and speaker but runs in the cloud. Amongst many goals, the first is to port little glados to a standalone piece of hardware, like an NVIDIA Jetson.

This is a proof of concept project, but a roadmap with farther goals lies below:

## Roadmap
Glados can:
- [X] Hold a conversation
- [ ] Act with a less corporate personality
- [ ] Filter out speech that isn't directed at Little Glados
- [ ] Call specified functions
- [ ] "Extend" prompts to accomplish tasks greater than one response   
- [ ] Run on standalone hardware (processor, speaker, microphone, case?, graphics?)
- [ ] Save and retrieve information from the system
- [ ] Semantic Analysis to judge emotion in the conversation
- [ ] Self-learn using data from regular operation


## To run:
1. Get **llama** access, this will be used as the "brain", it might take some time for them to send you the email with the link.
2. Get a [lambda instance](https://lambdalabs.com/) and throw one up, optionally set up a **filesystem** and attach it so you can come back later.
3. Navigate to the **filesystem** and clone this repo, then use your **llama** link to install the **llama-2-7b-chat** model by running `bash ./models/llama/download.sh`
4. Once installed run `pip install -r requirements.txt` to install the python libraries
5. Clone the repo on whatever computer you're going to run little glados on, `pip install -r requirements.txt` on there as well
6. Replace the `HOSTNAME` in both `glados.py` and `server.py` with the ip address of your **lambda** instance
7. run `run_server.sh` on the **lambda** instance and once it's ready run `glados.py` locally

