"""
Flask wrapper around TensorFlow Serving.

Responsibilities:
- Starts tensorflow_model_server in the background (gRPC :8500, REST :8501)
- Exposes a /monitoring endpoint with model health/status info
- Proxies all other requests to TF Serving's REST API on localhost:8501
- Listens on PORT (default 8080) so Railway's service domain resolves correctly
"""

import os
import subprocess
import sys
import time
import threading

import requests
from flask import Flask, jsonify, request, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", 8080))
MODEL_NAME = os.environ.get("MODEL_NAME", "andyid-model")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/models")
MONITORING_CONFIG = os.environ.get("MONITORING_CONFIG", "/model_config/prometheus.config")

TF_GRPC_PORT = 8500
TF_REST_PORT = 8501
TF_REST_BASE = f"http://localhost:{TF_REST_PORT}"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# TensorFlow Serving process management
# ---------------------------------------------------------------------------

def start_tf_serving():
    """Launch tensorflow_model_server as a background subprocess."""
    cmd = [
        "tensorflow_model_server",
        f"--port={TF_GRPC_PORT}",
        f"--rest_api_port={TF_REST_PORT}",
        f"--model_name={MODEL_NAME}",
        f"--model_base_path={MODEL_BASE_PATH}/{MODEL_NAME}",
        f"--monitoring_config_file={MONITORING_CONFIG}",
    ]
    print(f"[app.py] Starting TF Serving: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc


def wait_for_tf_serving(timeout: int = 60):
    """Block until TF Serving's REST API is reachable or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{TF_REST_BASE}/v1/models/{MODEL_NAME}", timeout=2)
            if r.status_code < 500:
                print("[app.py] TF Serving is ready.", flush=True)
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("[app.py] WARNING: TF Serving did not become ready within timeout.", flush=True)
    return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/monitoring", methods=["GET"])
def monitoring():
    """
    Health / status endpoint expected by the application.

    Returns basic model metadata plus the live model status from TF Serving
    when available.
    """
    status_info = {
        "model_name": MODEL_NAME,
        "model_base_path": f"{MODEL_BASE_PATH}/{MODEL_NAME}",
        "tf_serving_rest_port": TF_REST_PORT,
        "tf_serving_grpc_port": TF_GRPC_PORT,
        "status": "unknown",
        "versions": [],
    }

    try:
        r = requests.get(f"{TF_REST_BASE}/v1/models/{MODEL_NAME}", timeout=5)
        if r.status_code == 200:
            tf_data = r.json()
            model_versions = tf_data.get("model_version_status", [])
            status_info["status"] = "ok"
            status_info["versions"] = [
                {
                    "version": v.get("version"),
                    "state": v.get("state"),
                    "health": v.get("status", {}).get("error_code", "OK"),
                }
                for v in model_versions
            ]
        else:
            status_info["status"] = "degraded"
            status_info["tf_serving_response_code"] = r.status_code
    except requests.exceptions.RequestException as exc:
        status_info["status"] = "unavailable"
        status_info["error"] = str(exc)

    http_status = 200 if status_info["status"] == "ok" else 503
    return jsonify(status_info), http_status


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe — always returns 200 if the wrapper is running."""
    return jsonify({"status": "alive"}), 200


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy(path):
    """Forward every other request to TF Serving's REST API."""
    target_url = f"{TF_REST_BASE}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string.decode()}"

    try:
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers={k: v for k, v in request.headers if k.lower() != "host"},
            data=request.get_data(),
            timeout=30,
            allow_redirects=False,
        )
        excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        headers = [
            (k, v) for k, v in resp.raw.headers.items()
            if k.lower() not in excluded_headers
        ]
        return Response(resp.content, status=resp.status_code, headers=headers)
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": "TF Serving unreachable", "detail": str(exc)}), 502


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tf_proc = start_tf_serving()

    # Wait for TF Serving in a background thread so Flask can start quickly
    # (Railway health checks need the port to open fast)
    threading.Thread(target=wait_for_tf_serving, daemon=True).start()

    print(f"[app.py] Flask wrapper listening on port {PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)

    # If Flask exits, terminate TF Serving too
    tf_proc.terminate()
    tf_proc.wait()
