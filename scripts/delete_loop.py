import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.getcwd())
from remote_exec.hyperstack import HyperstackClient

load_dotenv('.env')
client = HyperstackClient(os.getenv('HYPERSTACK_API_KEY'))

vm_id = 953121
while True:
    try:
        data = client.get_vm(vm_id)
        vm = data.get("virtual_machine", data.get("instance", data))
        status = vm.get("status", "UNKNOWN").upper()
        print(f"Status: {status}")
        if status in ["ACTIVE", "SHUTOFF", "HIBERNATED", "ERROR"]:
            client.delete_vm(vm_id)
            print("Deleted successfully!")
            break
        time.sleep(10)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
