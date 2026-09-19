"""
Deployment and Tunneling Manager for Multi-Agent RAG Decision System
Launches the Streamlit Application and binds a public HTTPS tunnel.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

def run():
    print("=" * 60)
    print("🚀 STARTING MULTI-AGENT RAG DECISION ENGINE DEPLOYMENT")
    print("=" * 60)
    
    python_cmd = str(PYTHON_EXE) if PYTHON_EXE.exists() else "python"
    
    # 1. Start Streamlit Server
    print("\n[1/2] Starting Streamlit Application on port 8501...")
    streamlit_proc = subprocess.Popen(
        [python_cmd, "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to initialize
    time.sleep(3)
    
    print("\n[2/2] Connecting Public Tunnel...")
    print("Local Access URL:   http://localhost:8501")
    print("Network Access URL: http://0.0.0.0:8501")
    print("-" * 60)
    
    # Try starting localtunnel via npx
    try:
        tunnel_proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", "8501"],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("Waiting for tunnel URL...")
        for line in tunnel_proc.stdout:
            if "your url is:" in line.lower() or "https://" in line:
                print(f"\n🌐 PUBLIC LIVE DEPLOYMENT URL: {line.strip()}")
                break
    except Exception as e:
        print(f"Note: Local tunnel auto-forwarding: {e}")
        print("You can access the live application directly at: http://localhost:8501")
        
    print("\nApplication is running live. Press Ctrl+C to stop.")
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        streamlit_proc.terminate()

if __name__ == "__main__":
    run()
