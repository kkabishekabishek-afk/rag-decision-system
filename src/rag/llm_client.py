import os
import json
import requests
import ollama

def get_secret(key_name):
    """Retrieve secret from Streamlit secrets, session_state, or environment variables."""
    # 1. Environment variable
    val = os.getenv(key_name)
    if val:
        return val.strip()
    
    # 2. Streamlit session_state or secrets (if running inside Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "session_state") and st.session_state.get(key_name):
            return str(st.session_state.get(key_name)).strip()
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    
    return None

def call_groq(prompt, api_key, temperature=0.0):
    """Call Groq Cloud API for ultra-fast Llama 3 inference."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        # Fallback to 8b instant model if rate-limited or 70b unavailable
        payload["model"] = "llama-3.1-8b-instant"
        fallback_resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if fallback_resp.status_code == 200:
            return fallback_resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"Groq API Error ({response.status_code}): {response.text}")

def call_gemini(prompt, api_key, temperature=0.0):
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature
        }
    }
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"Gemini API Error ({response.status_code}): {response.text}")

def call_openai(prompt, api_key, temperature=0.0):
    """Call OpenAI compatible API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenAI API Error ({response.status_code}): {response.text}")

def call_llm(prompt, model="llama3.2:latest", temperature=0.0):
    """
    Universal LLM Dispatcher:
    1. Tries local Ollama first (if accessible).
    2. If Ollama fails (e.g. running on Streamlit Cloud), falls back seamlessly to
       Groq (Free Llama 3.3/3.2), Gemini, or OpenAI API key from secrets/sidebar.
    """
    # Check if Cloud LLM is forced via secret or if Ollama is running
    groq_key = get_secret("GROQ_API_KEY")
    gemini_key = get_secret("GEMINI_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")
    
    # 1. Try Ollama (Local)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response["message"]["content"]
    except Exception as ollama_err:
        # 2. Cloud Fallback if Ollama unreachable
        if groq_key:
            return call_groq(prompt, groq_key, temperature)
        elif gemini_key:
            return call_gemini(prompt, gemini_key, temperature)
        elif openai_key:
            return call_openai(prompt, openai_key, temperature)
        else:
            raise ConnectionError(
                "❌ Cloud Deployment Detected: Local Ollama is not accessible on Streamlit Cloud.\n\n"
                "👉 Solution: Please provide a FREE Groq API Key (runs Llama 3.3 for free) or Gemini API Key.\n"
                "You can enter it in the sidebar on your app, or add `GROQ_API_KEY = \"gsk_...\"` in your Streamlit Cloud App Settings ➔ Secrets.\n"
                "Get a free instant key at: https://console.groq.com/keys"
            )
