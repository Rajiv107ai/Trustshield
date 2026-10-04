"""Live end-to-end smoke test for TrustShield AI FastAPI server.

Starts the uvicorn server on a test port, queries all 4 endpoints over real HTTP:
  1. GET  /health
  2. POST /transaction/score
  3. GET  /fraud-rings
  4. POST /listing/analyze
Validates all responses and cleanly shuts down.
"""

import json
import socket
import subprocess
import sys
import time
import requests


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"[*] Starting uvicorn server on port {port}...")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # 1. Wait for server to become healthy
        print("[*] Waiting for server initialization...")
        healthy = False
        health_data = None
        for _ in range(30):
            try:
                resp = requests.get(f"{base_url}/health", timeout=2)
                if resp.status_code == 200:
                    health_data = resp.json()
                    if health_data.get("models_loaded"):
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(0.5)

        if not healthy:
            print("[-] Server failed to become healthy in time.")
            stdout, stderr = proc.communicate(timeout=2)
            print(f"Stdout:\n{stdout}")
            print(f"Stderr:\n{stderr}")
            sys.exit(1)

        print("[+] Endpoint 1: GET /health SUCCESS")
        print(json.dumps(health_data, indent=2))

        # 2. Test /transaction/score
        tx_payload = {
            "order_id": "ORD_LIVE_001",
            "buyer_id": "BUYER_000001",
            "seller_id": "SELLER_000001",
            "amount": 250.0,
            "base_price": 250.0,
            "category_median_price": 200.0,
            "order_date": "2025-09-01T12:00:00",
            "buyer_orders_before": 15,
            "buyer_returns_before": 2,
        }
        resp = requests.post(f"{base_url}/transaction/score", json=tx_payload, timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        tx_data = resp.json()
        print("\n[+] Endpoint 2: POST /transaction/score SUCCESS")
        print(json.dumps(tx_data, indent=2))
        assert "overall_fraud_probability" in tx_data
        assert "risk_label" in tx_data
        assert tx_data["order_id"] == "ORD_LIVE_001"

        # 3. Test /fraud-rings
        resp = requests.get(f"{base_url}/fraud-rings?limit=3", timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        rings_data = resp.json()
        print("\n[+] Endpoint 3: GET /fraud-rings SUCCESS")
        print(json.dumps(rings_data, indent=2))
        assert "rings" in rings_data
        assert rings_data["total_rings"] > 0

        # 4. Test /listing/analyze
        listing_payload = {
            "listing_id": "LST_LIVE_999",
            "seller_id": "SELLER_SUSPECT_01",
            "price": 29.99,
            "base_price": 199.99,
            "category_median_price": 180.0,
            "seller_age_days_at_listing": 3.0,
            "seller_listings_before": 2,
            "multimodal_similarity_score": 0.15,
        }
        resp = requests.post(f"{base_url}/listing/analyze", json=listing_payload, timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        listing_data = resp.json()
        print("\n[+] Endpoint 4: POST /listing/analyze SUCCESS")
        print(json.dumps(listing_data, indent=2))
        assert "fake_listing_probability" in listing_data
        assert "investigator_narrative" in listing_data
        assert listing_data["listing_id"] == "LST_LIVE_999"

        print("\n==================================================")
        print("ALL 4 API ENDPOINTS VERIFIED END-TO-END SUCCESSFULLY!")
        print("==================================================")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
