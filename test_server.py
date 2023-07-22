import requests

res = requests.post("https://localhost:8080", data="hello", verify=False)
print(res.text)