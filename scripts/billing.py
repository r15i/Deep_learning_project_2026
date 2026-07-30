import os, requests
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('HYPERSTACK_API_KEY='):
            key = line.split('=', 1)[1].strip()
            break
print(requests.get('https://infrahub-api.nexgencloud.com/v1/billing/profile', headers={'Authorization': f'Bearer {key}'}).json())
