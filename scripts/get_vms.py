import requests, os
API_KEY = os.environ.get("HYPERSTACK_API_KEY", "")
headers = {"Authorization": f"Bearer {API_KEY}"}
r = requests.get("https://infrahub-api.nexgencloud.com/v1/core/virtual-machines", headers=headers)
print(r.json())
