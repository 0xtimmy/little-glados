import requests

res = requests.post("https://104.171.202.180:8000", data="hello", verify=False)
print(res.text)