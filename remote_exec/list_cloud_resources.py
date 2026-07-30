#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List all available Hyperstack cloud images, VM flavors, and pricing.

Usage:
    python3 list_cloud_resources.py          # show everything
    python3 list_cloud_resources.py --gpu    # GPU flavors only
    python3 list_cloud_resources.py --cpu    # CPU flavors only
"""

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from hyperstack import HyperstackClient

import requests

# ─────────────────────────────────────────────────────────────────────
# Known GPU specifications (CUDA cores and TDP in watts)
# ─────────────────────────────────────────────────────────────────────
GPU_SPECS = {
    # name-in-pricebook      CUDA cores   TDP (W)
    "RTX-A4000":              (6144,       140),
    "RTX-A5000":              (8192,       230),
    "RTX-A6000":              (10752,      300),
    "RTX-A6000-ada":          (18176,      300),
    "RTX-A6000-spot":         (10752,      300),
    "A40":                    (10752,      300),
    "A100-80G-PCIe":          (6912,       300),
    "A100-80G-PCIe-spot":     (6912,       300),
    "A100-80G-PCIe-NVLink":   (6912,       400),
    "A100-80G-SXM4":          (6912,       400),
    "L40":                    (18176,      300),
    "L40-spot":               (18176,      300),
    "H100-80G-PCIe":          (16896,      350),
    "H100-80G-PCIe-spot":     (16896,      350),
    "H100-80G-PCIe-NVLink":   (16896,      350),
    "H100-80G-SXM5":          (16896,      700),
    "H200-141G-SXM5":         (16896,      700),
    "B200-SXM":               (18432,      1000),
    "RTX-4090":               (16384,      450),
    "RTX-5090":               (21760,      575),
    "RTX-PRO6000-SE":         (18176,      250),
    "RTX-PRO6000-SE-spot":    (18176,      250),
}


def fetch_pricebook(api_key: str) -> dict:
    """Return {resource_name: hourly_price_float} from the pricebook API."""
    headers = {"api_key": api_key, "Content-Type": "application/json"}
    r = requests.get("https://infrahub-api.nexgencloud.com/v1/pricebook", headers=headers)
    r.raise_for_status()
    return {item["name"]: float(Decimal(item["value"])) for item in r.json()}


def compute_flavor_price(flav: dict, prices: dict) -> float | None:
    """Compute the total hourly price for a flavor."""
    gpu_name = flav.get("gpu", "")
    gpu_count = flav.get("gpu_count", 0)
    cpu_count = flav.get("cpu", 0)
    ram_gb = flav.get("ram", 0)
    disk_gb = flav.get("disk", 0)
    ephemeral_gb = flav.get("ephemeral", 0)

    total = 0.0

    if gpu_count > 0 and gpu_name:
        gpu_price = prices.get(gpu_name)
        if gpu_price is None:
            return None  # unknown GPU pricing
        total += gpu_price * gpu_count
        # vCPU and RAM are free for GPU flavors
    else:
        # CPU-only flavors
        vcpu_price = prices.get("vCPU (cpu-only-flavors)", 0)
        ram_price = prices.get("RAM (cpu-only-flavors)", 0)
        total += vcpu_price * cpu_count
        total += ram_price * ram_gb

    # Storage cost
    ssd_price = prices.get("Cloud-SSD", 0)
    total += ssd_price * disk_gb

    return total


def main():
    parser = argparse.ArgumentParser(description="List Hyperstack cloud resources and pricing")
    parser.add_argument("--gpu", action="store_true", help="Show only GPU flavors")
    parser.add_argument("--cpu", action="store_true", help="Show only CPU flavors")
    args = parser.parse_args()

    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    api_key = os.getenv("HYPERSTACK_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("ERROR: HYPERSTACK_API_KEY is not set in .env")
        sys.exit(1)

    client = HyperstackClient(api_key)

    # ── Fetch pricing ─────────────────────────────────────────────────
    print("Fetching pricebook...")
    prices = fetch_pricebook(api_key)

    # ── Images ────────────────────────────────────────────────────────
    if not args.gpu and not args.cpu:
        print("\n" + "=" * 80)
        print("  OS IMAGES")
        print("=" * 80)
        try:
            images_resp = client.list_images()
            # Deduplicate by name
            seen = set()
            print(f"\n  {'Name':<55} | {'Region':<12} | {'Type':<10}")
            print("  " + "-" * 82)
            for group in images_resp.get("images", []):
                region = group.get("region_name", "")
                for img in group.get("images", []):
                    name = img.get("name", "")
                    key = (name, region)
                    if key in seen:
                        continue
                    seen.add(key)
                    print(f"  {name[:54]:<55} | {region[:11]:<12} | {img.get('type', '')[:9]:<10}")
        except Exception as e:
            print(f"  Error fetching images: {e}")

    # ── Flavors ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)

    if args.cpu:
        print("  CPU-ONLY FLAVORS (no GPU)")
    elif args.gpu:
        print("  GPU FLAVORS")
    else:
        print("  VM FLAVORS + PRICING")
    print("=" * 80)

    try:
        flavors_resp = client.list_flavors()

        # Flatten nested structure
        flavors_data = flavors_resp.get("data", flavors_resp.get("flavors", []))
        all_flavors = []
        for f_item in flavors_data:
            if "flavors" in f_item:
                all_flavors.extend(f_item["flavors"])
            else:
                all_flavors.append(f_item)

        # Filter
        if args.cpu:
            all_flavors = [f for f in all_flavors if f.get("gpu_count", 0) == 0]
        elif args.gpu:
            all_flavors = [f for f in all_flavors if f.get("gpu_count", 0) > 0]

        # Deduplicate by name (keep first occurrence)
        seen_names = set()
        unique_flavors = []
        for f in all_flavors:
            name = f.get("name", "")
            if name not in seen_names:
                seen_names.add(name)
                unique_flavors.append(f)
        all_flavors = unique_flavors

        # ── CPU-only table ────────────────────────────────────────────
        cpu_flavors = [f for f in all_flavors if f.get("gpu_count", 0) == 0]
        gpu_flavors = [f for f in all_flavors if f.get("gpu_count", 0) > 0]

        if cpu_flavors and not args.gpu:
            print(f"\n  {'Name':<22} | {'CPU':<4} | {'RAM':<6} | {'Disk':<6} | {'$/hr':<8} | {'Stock':<5}")
            print("  " + "-" * 68)
            for flav in cpu_flavors:
                price = compute_flavor_price(flav, prices)
                price_str = f"${price:.3f}" if price is not None else "N/A"
                stock = "✓" if flav.get("stock_available") else "✗"
                print(f"  {flav['name'][:21]:<22} | {flav.get('cpu', ''):<4} | {flav.get('ram', 0):<6} | {flav.get('disk', 0):<6} | {price_str:<8} | {stock:<5}")

        # ── GPU table ─────────────────────────────────────────────────
        if gpu_flavors and not args.cpu:
            if cpu_flavors and not args.gpu:
                print()  # separator between sections

            header = (
                f"  {'Name':<32} | {'GPU':<24} | {'CPU':<4} | {'RAM':<7} | {'Disk':<5} | "
                f"{'$/hr':<8} | {'CUDA':<7} | {'TDP(W)':<6} | {'CUDA/$':<8} | {'CUDA/W':<7} | {'Stock'}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))

            for flav in gpu_flavors:
                gpu_name = flav.get("gpu", "")
                gpu_count = flav.get("gpu_count", 0)
                gpu_label = f"{gpu_count}x {gpu_name}" if gpu_count > 0 else "None"

                price = compute_flavor_price(flav, prices)
                price_str = f"${price:.2f}" if price is not None else "N/A"

                # GPU specs lookup
                specs = GPU_SPECS.get(gpu_name)
                if specs:
                    cuda_per_gpu, tdp_per_gpu = specs
                    total_cuda = cuda_per_gpu * gpu_count
                    total_tdp = tdp_per_gpu * gpu_count
                    cuda_str = f"{total_cuda:,}"
                    tdp_str = f"{total_tdp}"

                    # Efficiency metrics
                    if price and price > 0:
                        cuda_per_dollar = total_cuda / price
                        cuda_dollar_str = f"{cuda_per_dollar:,.0f}"
                    else:
                        cuda_dollar_str = "N/A"

                    if total_tdp > 0:
                        cuda_per_watt = total_cuda / total_tdp
                        cuda_watt_str = f"{cuda_per_watt:.1f}"
                    else:
                        cuda_watt_str = "N/A"
                else:
                    cuda_str = "?"
                    tdp_str = "?"
                    cuda_dollar_str = "?"
                    cuda_watt_str = "?"

                stock = "✓" if flav.get("stock_available") else "✗"

                print(
                    f"  {flav['name'][:31]:<32} | {gpu_label[:23]:<24} | {flav.get('cpu', ''):<4} | "
                    f"{flav.get('ram', 0):<7} | {flav.get('disk', 0):<5} | {price_str:<8} | "
                    f"{cuda_str:<7} | {tdp_str:<6} | {cuda_dollar_str:<8} | {cuda_watt_str:<7} | {stock}"
                )

    except Exception as e:
        print(f"  Error fetching flavors: {e}")

    # ── Pricebook summary ─────────────────────────────────────────────
    if not args.gpu and not args.cpu:
        print("\n" + "=" * 80)
        print("  RAW PRICEBOOK (per-unit hourly rates)")
        print("=" * 80)
        # Only show hardware-related prices (skip LLM inference pricing)
        hw_prices = {k: v for k, v in prices.items() if "/" not in k}
        print(f"\n  {'Resource':<35} | {'$/hr':<12}")
        print("  " + "-" * 50)
        for name, val in sorted(hw_prices.items()):
            print(f"  {name:<35} | ${val:.6f}")


if __name__ == "__main__":
    main()
