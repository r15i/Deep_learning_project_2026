#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperstack REST API client.

Thin wrapper around the Hyperstack (infrahub) REST API.
Docs: https://docs.hyperstack.cloud/docs/api-reference/introduction/

All methods return the parsed JSON response dict.
"""

import time
import requests

BASE_URL = "https://infrahub-api.nexgencloud.com/v1"


class HyperstackClient:
    """Stateless HTTP client for the Hyperstack REST API."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "api_key": api_key,
            "Content-Type": "application/json",
        })

    # ── helpers ────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{BASE_URL}{path}"

    def _get(self, path: str, **kwargs) -> dict:
        r = self.session.get(self._url(path), **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json_body: dict) -> dict:
        r = self.session.post(self._url(path), json=json_body)
        print(r.text)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = self.session.delete(self._url(path))
        print(r.text)
        r.raise_for_status()
        return r.json()

    # ── Environments ──────────────────────────────────────────────────

    def list_environments(self) -> dict:
        return self._get("/core/environments")

    # ── Flavors ───────────────────────────────────────────────────────

    def list_flavors(self) -> dict:
        return self._get("/core/flavors")

    # ── Images ────────────────────────────────────────────────────────

    def list_images(self) -> dict:
        return self._get("/core/images")

    # ── Key-pairs ─────────────────────────────────────────────────────

    def list_keypairs(self) -> dict:
        return self._get("/core/keypairs")

    def create_keypair(self, name: str, public_key: str, environment_name: str) -> dict:
        return self._post("/core/keypairs", {
            "name": name,
            "public_key": public_key,
            "environment_name": environment_name,
        })

    def delete_keypair(self, keypair_name: str) -> dict:
        return self._delete(f"/core/keypairs/{keypair_name}")

    # ── Virtual Machines ──────────────────────────────────────────────

    def create_vm(self, name: str, environment_name: str,
                  flavor_name: str, key_name: str,
                  image_name: str, user_data: str = "",
                  count: int = 1,
                  assign_floating_ip: bool = True,
                  callback_url: str = "") -> dict:
        body = {
            "name": name,
            "environment_name": environment_name,
            "flavor_name": flavor_name,
            "key_name": key_name,
            "image_name": image_name,
            "count": count,
            "assign_floating_ip": assign_floating_ip,
        }
        if user_data:
            body["user_data"] = user_data
        if callback_url:
            body["callback_url"] = callback_url
        return self._post("/core/virtual-machines", body)

    def list_vms(self) -> dict:
        return self._get("/core/virtual-machines")

    def get_vm(self, vm_id: int) -> dict:
        return self._get(f"/core/virtual-machines/{vm_id}")

    def delete_vm(self, vm_id: int) -> dict:
        return self._delete(f"/core/virtual-machines/{vm_id}")

    def create_vm_sg_rule(self, vm_id: int, protocol: str, port_range_min: int, port_range_max: int, direction: str = "ingress", remote_ip_prefix: str = "0.0.0.0/0", ethertype: str = "IPv4") -> dict:
        body = {
            "protocol": protocol,
            "port_range_min": port_range_min,
            "port_range_max": port_range_max,
            "direction": direction,
            "remote_ip_prefix": remote_ip_prefix,
            "ethertype": ethertype,
        }
        return self._post(f"/core/virtual-machines/{vm_id}/sg-rules", body)

    # ── Convenience ───────────────────────────────────────────────────

    def wait_for_vm(self, vm_id: int, target_status: str = "ACTIVE",
                    timeout: int = 600, poll_interval: int = 15) -> dict:
        """Poll until VM reaches *target_status* or *timeout* seconds elapse."""
        import sys
        deadline = time.time() + timeout
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while time.time() < deadline:
            data = self.get_vm(vm_id)
            vm = data.get("virtual_machine", data.get("instance", data))
            status = vm.get("status", "UNKNOWN")
            status_str = str(status).upper()
            sys.stdout.write(f"\r  [{spinner[idx]}] VM {vm_id} status: {status_str}    ")
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
            if status_str == target_status.upper():
                print()
                return data
            time.sleep(poll_interval)
        print()
        raise TimeoutError(
            f"VM {vm_id} did not reach '{target_status}' within {timeout}s"
        )

    def get_vm_ip(self, vm_data: dict) -> str | None:
        """Extract the first floating IP from a VM response dict."""
        vm = vm_data.get("virtual_machine", vm_data.get("instance", vm_data))
        # Try floating IPs first, then fixed IPs
        for key in ("floating_ip", "floating_ips", "fixed_ips"):
            val = vm.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, list) and val:
                ip = val[0]
                return ip.get("ip_address", ip) if isinstance(ip, dict) else str(ip)
        return None
