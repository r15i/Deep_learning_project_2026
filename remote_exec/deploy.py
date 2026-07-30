#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy.py — Provision machines, run training/tests, and tear down.

Manages Hyperstack via a local JSON state file.
Reuses Docker containers for faster sequential job execution.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from hyperstack import HyperstackClient

# ───────────────────────────────────────────────────────────────────────
# Config & State
# ───────────────────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).parent.parent / os.environ.get("DEPLOY_STATE_FILE", "remote_state.json")
ENV_FILE = Path(__file__).parent.parent / ".env"

def load_config() -> dict:
    if not ENV_FILE.exists():
        print(f"ERROR: {ENV_FILE} not found. Copy .env.example → .env")
        sys.exit(1)
    load_dotenv(ENV_FILE)
    
    required = [
        "HYPERSTACK_API_KEY", "HYPERSTACK_ENVIRONMENT",
        "HYPERSTACK_FLAVOR", "HYPERSTACK_IMAGE", "HYPERSTACK_KEY_NAME",
    ]
    cfg = {}
    for key in required:
        cfg[key] = os.getenv(key, "")
    
    cfg["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    cfg["SSH_PRIVATE_KEY_PATH"] = os.getenv("SSH_PRIVATE_KEY_PATH", "~/.ssh/id_rsa")
    cfg["DOCKER_IMAGE"] = os.getenv("DOCKER_IMAGE", "docker.io/r15i/nndl-project:latest")
    return cfg

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
            
    # Default state if not found
    return {"machines": {}}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ───────────────────────────────────────────────────────────────────────
# SSH Helpers
# ───────────────────────────────────────────────────────────────────────

def build_ssh_cmd(mconf: dict, command: str, tty=False) -> list:
    base = []
    if mconf.get("ssh_pass"):
        base.extend(["sshpass", "-p", mconf["ssh_pass"]])
    
    ssh_opts = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-p", str(mconf.get("port", 22))]
    if tty:
        ssh_opts.append("-t")
        
    base.extend(ssh_opts)
    if mconf.get("ssh_key_path"):
        base.extend(["-i", os.path.expanduser(mconf["ssh_key_path"])])
    base.append(f"{mconf['user']}@{mconf['ip']}")
    base.append(command)
    return base

def ssh_exec(mconf: dict, command: str, stream=False, check=False):
    cmd = build_ssh_cmd(mconf, command, tty=stream)
    if stream:
        return subprocess.run(cmd, check=check)
    else:
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

def scp_download(mconf: dict, remote_path: str, local_path: str):
    os.makedirs(local_path, exist_ok=True)
    rsync_cmd = ["rsync", "-avz", "--update"]
    if mconf.get("ssh_pass"):
        rsync_cmd = ["sshpass", "-p", mconf["ssh_pass"]] + rsync_cmd
    
    ssh_opts = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {mconf.get('port', 22)}"
    if mconf.get("ssh_key_path"):
        ssh_opts += f" -i {os.path.expanduser(mconf['ssh_key_path'])}"
        
    rsync_cmd.extend(["-e", ssh_opts])
    
    # Let rsync handle standard trailing slash directory nesting semantics natively
        
    rsync_cmd.extend([f"{mconf['user']}@{mconf['ip']}:{remote_path}", local_path])
    subprocess.run(rsync_cmd, check=True)


DOCKER_CLOUD_INIT = """#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/nndl_cloud_init.log) 2>&1
echo "=== Installing Docker ==="
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker
echo "=== Installing NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit wireguard openresolv
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "=== Setup complete ==="
"""

# ───────────────────────────────────────────────────────────────────────
# Provisioning & Execution
# ───────────────────────────────────────────────────────────────────────

def check_online(ip: str, port: int, timeout: int = 3) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False

def ensure_machine(name: str, cfg: dict, state: dict) -> dict:
    if name in state["machines"]:
        m = state["machines"][name]
        if not check_online(m["ip"], m.get("port", 22)):
            print(f"Machine '{name}' ({m['ip']}) is unresponsive! Removing from state to reprovision...")
            del state["machines"][name]
            save_state(state)
        else:
            print(f"Machine '{name}' is online and verified.")

    if name not in state["machines"]:
        print(f"Machine '{name}' not in state. Provisioning via Hyperstack...")
        client = HyperstackClient(cfg["HYPERSTACK_API_KEY"])
        
        image = cfg["HYPERSTACK_IMAGE"]
        user_data = "#!/bin/bash\nexec > >(tee /var/log/nndl_cloud_init.log) 2>&1\necho 'Docker already installed! Skipping...'\n" if "with Docker" in image else DOCKER_CLOUD_INIT

        import uuid
        res = client.create_vm(
            name=f"{name}-{uuid.uuid4().hex[:4]}",
            environment_name=cfg["HYPERSTACK_ENVIRONMENT"],
            flavor_name=cfg["HYPERSTACK_FLAVOR"],
            key_name=cfg["HYPERSTACK_KEY_NAME"],
            image_name=image,
            user_data=user_data,
            assign_floating_ip=True,
        )
        vms = res.get("virtual_machines", res.get("instances", [res]))
        vm = vms[0] if isinstance(vms, list) and vms else res.get("virtual_machine", res.get("instance", res))
        vm_id = vm.get("id")
        print(f"Created VM ID {vm_id}. Waiting to become ACTIVE...")
        
        data = client.wait_for_vm(vm_id, timeout=600, poll_interval=20)
        
        ip = None
        for _ in range(15):
            ip = client.get_vm_ip(data)
            if ip: break
            time.sleep(10)
            data = client.get_vm(vm_id)
            
        if not ip:
            print("ERROR: VM active but no floating IP attached.")
            sys.exit(1)
            
        try:
            client.create_vm_sg_rule(vm_id, protocol="tcp", port_range_min=22, port_range_max=22)
            client.create_vm_sg_rule(vm_id, protocol="tcp", port_range_min=3000, port_range_max=3000)
        except Exception as e:
            print(f"Warning setting SG rules: {e}")

        mconf = {
            "type": "hyperstack",
            "ip": ip,
            "vm_id": vm_id,
            "user": "ubuntu",
            "port": 22,
            "ssh_key_path": cfg["SSH_PRIVATE_KEY_PATH"],
            "container_name": "nndl_worker"
        }
        state["machines"][name] = mconf
        save_state(state)
        
        # Wait for SSH
        print("Waiting for SSH to be ready...")
        for _ in range(30):
            r = ssh_exec(mconf, "echo SSH_OK")
            if r.returncode == 0 and "SSH_OK" in r.stdout:
                print("SSH is ready.")
                break
            time.sleep(20)
            
        print("Waiting for cloud-init to finish (Docker & NVIDIA)...")
        for _ in range(40):
            r = ssh_exec(mconf, "cloud-init status")
            if r.returncode == 0 and "status: done" in r.stdout:
                print("Cloud-init done.")
                break
            time.sleep(20)
            
    return state["machines"][name]

def ensure_container(mconf: dict, cfg: dict):
    print(f"Checking if docker container '{mconf['container_name']}' is up on {mconf['type']} machine...")
    cname = mconf["container_name"]
    res = ssh_exec(mconf, f"docker ps -q --filter name={cname}")
    if res.returncode != 0:
        # Fallback to sudo docker
        res = ssh_exec(mconf, f"sudo docker ps -q --filter name={cname}")
        
    if res.stdout and res.stdout.strip():
        print(f"Container '{cname}' is already running. Reusing it.")
        return

    print(f"Container not running. Spinning up '{cname}'...")
    image = cfg["DOCKER_IMAGE"]
    wandb_key = cfg.get("WANDB_API_KEY", "")
    
    setup_cmd = f"""bash -c '
    if docker ps >/dev/null 2>&1; then DOCKER_CMD="docker"; else DOCKER_CMD="sudo docker"; fi
    echo "Pulling {image}..."
    $DOCKER_CMD pull {image}
    $DOCKER_CMD rm -f {cname} 2>/dev/null || true
    
    echo "Starting Dockhand..."
    $DOCKER_CMD rm -f dockhand 2>/dev/null || true
    $DOCKER_CMD run -d --name dockhand --restart unless-stopped -p 3000:3000 -v /var/run/docker.sock:/var/run/docker.sock -v dockhand_data:/app/data fnsys/dockhand:latest >/dev/null
    
    mkdir -p $HOME/nndl_weights $HOME/dataset $HOME/.cache/uv
    
    echo "Starting {cname} in background (sleep infinity)..."
    $DOCKER_CMD run -d \
        --name {cname} \
        --hostname {mconf.get("type","worker")} \
        --network host \
        --ipc=host \
        --privileged \
        --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e CUDA_VISIBLE_DEVICES=0 \
        -v $HOME/nndl_weights:/app/weights \
        -v $HOME/dataset:/app/dataset \
        -v $HOME/.cache/uv:/root/.cache/uv \
        -e WANDB_API_KEY="{wandb_key}" \
        {image} \
        sleep infinity
    '"""
    res = ssh_exec(mconf, setup_cmd)
    if res.returncode != 0:
        print("ERROR setting up container:\n", res.stderr)
        sys.exit(1)
    print("Container is up.")

# ───────────────────────────────────────────────────────────────────────
# Commands
# ───────────────────────────────────────────────────────────────────────

def cmd_execute(args, cfg, state):
    mconf = ensure_machine(args.machine, cfg, state)
    ensure_container(mconf, cfg)
    
    cname = mconf["container_name"]
    target = args.target
    make_args = " ".join(args.extra_args)
    
    print(f"\nExecuting: make {target} {make_args} on {args.machine}")
    
    # Sync local code to remote container to ensure hot-patches take effect
    print("Syncing local code to remote container...")
    
    # Create temp dir on remote
    ssh_exec(mconf, "mkdir -p /tmp/nndl_sync")
    
    # Build rsync command
    rsync_cmd = ["rsync", "-avz", "--exclude", ".git", "--exclude", "weights", "--exclude", "dataset", "--exclude", ".venv", "./", f"{mconf['user']}@{mconf['ip']}:/tmp/nndl_sync/"]
    if mconf.get("ssh_pass"):
        rsync_cmd = ["sshpass", "-p", mconf["ssh_pass"]] + rsync_cmd
    
    import subprocess
    subprocess.run(rsync_cmd, env=dict(os.environ, RSYNC_RSH=f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {mconf.get('port', 22)}"), check=False)
    
    dcmd = "sudo docker" if "sudo" in ssh_exec(mconf, "sudo docker ps >/dev/null && echo sudo").stdout else "docker"
    
    # Copy from remote host to container
    ssh_exec(mconf, f"{dcmd} cp /tmp/nndl_sync/. {cname}:/app/")
    
    # Execute the make command interactively
    cmd = f"{dcmd} exec {cname} bash -c 'cd /app && /app/scripts/bootstrap.sh make {target} {make_args}'"
    
    res = ssh_exec(mconf, cmd, stream=True)
    if res.returncode != 0:
        print(f"\nWARNING: Command exited with code {res.returncode}")
        sys.exit(res.returncode)
    else:
        print("\n✓ Execution complete.")

def cmd_download(args, cfg, state):
    if args.machine not in state["machines"]:
        print(f"Machine {args.machine} not found in state.")
        sys.exit(1)
    mconf = state["machines"][args.machine]
    dest = Path(args.dest)
    print(f"Downloading from {args.machine} to {dest}...")
    
    base_remote = f"/home/{mconf.get('user', 'ubuntu')}/nndl_weights/"
    
    if hasattr(args, "subpath") and args.subpath:
        remote_path = base_remote.rstrip('/') + '/' + args.subpath.strip('/')
        local_path = str(dest)
    else:
        # Default behavior: mirror only the machine's specific environment directory
        remote_path = base_remote.rstrip('/') + '/' + args.machine
        local_path = str(dest)
        
    scp_download(mconf, remote_path, local_path)
    print(f"✓ Downloaded to {local_path}")


def cmd_clean_container(args, cfg, state):
    if args.machine not in state["machines"]:
        print(f"Machine {args.machine} not found in state.")
        sys.exit(1)
    mconf = state["machines"][args.machine]
    cname = mconf["container_name"]
    print(f"Cleaning container '{cname}' on {args.machine}...")
    dcmd = "sudo docker" if "sudo" in ssh_exec(mconf, "sudo docker ps >/dev/null && echo sudo").stdout else "docker"
    ssh_exec(mconf, f"{dcmd} rm -f {cname}")
    print("Container cleaned.")

def cmd_clean_machine(args, cfg, state):
    if args.machine not in state["machines"]:
        print(f"Machine {args.machine} not found in state.")
        sys.exit(1)
        
    mconf = state["machines"][args.machine]
    if mconf["type"] == "hyperstack":
        print(f"Destroying Hyperstack VM for {args.machine}...")
        client = HyperstackClient(cfg["HYPERSTACK_API_KEY"])
        client.delete_vm(mconf["vm_id"])
        print("VM destroyed.")
        
    del state["machines"][args.machine]
    save_state(state)
    print(f"Machine {args.machine} removed from state.")


# Hyperstack General Commands
def cmd_list_flavors(cfg, client):
    data = client.list_flavors()
    flavors = data.get("flavors", data)
    if isinstance(flavors, list):
        for f in flavors:
            print(f"  {f.get('name', 'N/A'):40s}  GPU: {f.get('gpu', 'N/A')}  CPU: {f.get('cpu', 'N/A')}  RAM: {f.get('ram', 'N/A')} GB")

def cmd_list_images(cfg, client):
    data = client.list_images()
    for group in data.get("images", []):
        region = group.get("region_name", "Unknown Region")
        print(f"  Region: {region}")
        for img in group.get("images", []):
            print(f"    - {img.get('name', 'N/A'):50s} type: {img.get('type', 'N/A')}")

def cmd_list_vms(cfg, client):
    data = client.list_vms()
    vms = data.get("virtual_machines", data.get("instances", []))
    print("  CURRENT VMs:")
    if isinstance(vms, list) and vms:
        for vm in vms:
            print(f"    - ID: {vm.get('id')} | Name: {vm.get('name')} | Status: {vm.get('status')} | Flavor: {vm.get('flavor_name')} | IP: {client.get_vm_ip(vm)}")
    else:
        print("    No VMs found.")

def cmd_list_envs(cfg, client):
    data = client.list_environments()
    envs = data.get("environments", [])
    print("  ENVIRONMENTS:")
    if isinstance(envs, list) and envs:
        for e in envs:
            print(f"    - Name: {e.get('name')} | Region: {e.get('region')} | Status: {e.get('status')}")
    else:
        print("    No environments found.")

def cmd_list_keys(cfg, client):
    data = client.list_keypairs()
    keys = data.get("keypairs", [])
    print("  SSH KEYPAIRS:")
    if isinstance(keys, list) and keys:
        for k in keys:
            print(f"    - Name: {k.get('name')} | Fingerprint: {k.get('fingerprint')} | Env: {k.get('environment_name')}")
    else:
        print("    No keys found.")

def cmd_state(cfg, client):
    print("=== HYPERSTACK GENERAL STATE ===")
    cmd_list_vms(cfg, client)
    print("")
    cmd_list_envs(cfg, client)
    print("")
    cmd_list_keys(cfg, client)


def main():
    p = argparse.ArgumentParser(
        description="Deploy & manage NNDL training jobs on Hyperstack VMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    
    c_exec = sub.add_parser("execute", help="Run a make target on a remote machine (spins up if needed)")
    c_exec.add_argument("--machine", required=True, help="Machine name (e.g., 4nt0n, hyperstack)")
    c_exec.add_argument("--target", required=True, help="Make target")
    c_exec.add_argument("extra_args", nargs="*", help="Extra args to pass to make")
    
    c_cc = sub.add_parser("clean-container", help="Destroy the docker container on the remote machine")
    c_cc.add_argument("--machine", required=True)
    
    c_cm = sub.add_parser("clean-machine", help="Destroy the remote machine entirely")
    c_cm.add_argument("--machine", required=True)

    c_dl = sub.add_parser("download", help="Download weights from the machine")
    c_dl.add_argument("--machine", required=True)
    c_dl.add_argument("--dest", default="./weights", help="Local dest dir")
    c_dl.add_argument("--subpath", default="", help="Subpath within remote nndl_weights to download (e.g., '4nt0n/test')")

    sub.add_parser("list-flavors", help="List available GPU flavors")
    sub.add_parser("list-images", help="List available OS images")
    sub.add_parser("list-vms", help="List current VMs/instances")
    sub.add_parser("list-envs", help="List available environments")
    sub.add_parser("list-keys", help="List SSH keypairs")
    sub.add_parser("state", help="Print current general state (VMs, Envs, Keys)")

    args = p.parse_args()
    cfg = load_config()
    state = load_state()
    
    if args.command == "execute":
        cmd_execute(args, cfg, state)
    elif args.command == "clean-container":
        cmd_clean_container(args, cfg, state)
    elif args.command == "clean-machine":
        cmd_clean_machine(args, cfg, state)
    elif args.command == "download":
        cmd_download(args, cfg, state)
    else:
        # Hyperstack general commands
        client = HyperstackClient(cfg["HYPERSTACK_API_KEY"])
        if args.command == "list-flavors":
            cmd_list_flavors(cfg, client)
        elif args.command == "list-images":
            cmd_list_images(cfg, client)
        elif args.command == "list-vms":
            cmd_list_vms(cfg, client)
        elif args.command == "list-envs":
            cmd_list_envs(cfg, client)
        elif args.command == "list-keys":
            cmd_list_keys(cfg, client)
        elif args.command == "state":
            cmd_state(cfg, client)


if __name__ == "__main__":
    main()
