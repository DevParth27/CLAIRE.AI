import requests

url = "https://2f62c21a7b17.ngrok-free.app"
headers = {
    "ngrok-skip-browser-warning": "yes"
}
response = requests.get(url, headers=headers)
print(response.text)