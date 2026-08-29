import requests

url = "http://localhost:8000/api/v1/predict"
file_path = r"D:\Client-projects\coffee-leaf-disease\data\raw\rocole\coffee___healthy\C10P10E1.jpg"

print(f"Uploading {file_path}...")
with open(file_path, "rb") as f:
    files = {"file": ("C10P10E1.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
try:
    print("Response:", response.json())
except:
    print("Text:", response.text)
