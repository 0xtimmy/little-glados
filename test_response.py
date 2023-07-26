
from pydub import AudioSegment
from pydub.playback import play
import requests

HOSTNAME = "104.171.203.39"

with open("speech.wav", "rb") as f:
    res = requests.post(f"https://{HOSTNAME}:8000", data=f.read(), verify=False)
    with open("response.wav", "w+b") as f:
        f.write(res.content)
        f.close()
    sound = AudioSegment.from_wav("response.wav")
    play(sound)