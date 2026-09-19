import os
import json
import time
import requests
import re

def get_secret(key_name):
    """Retrieve secret from Streamlit secrets, session_state, or environment variables."""
    val = os.getenv(key_name)
    if val:
        return val.strip()
    
    try:
        import streamlit as st
        if hasattr(st, "session_state") and st.session_state.get(key_name):
            return str(st.session_state.get(key_name)).strip()
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    
    return None

def call_ollama(prompt, model="llama3.2:latest", temperature=0.0):
    """Try calling local Ollama server."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )
    return response["message"]["content"]

def call_groq(prompt, api_key, temperature=0.0):
    """Call Groq Cloud API."""
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
    response = requests.post(url, headers=headers, json=payload, timeout=40)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        payload["model"] = "llama-3.1-8b-instant"
        fallback_resp = requests.post(url, headers=headers, json=payload, timeout=40)
        if fallback_resp.status_code == 200:
            return fallback_resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"Groq API Error: {response.text}")

def call_gemini(prompt, api_key, temperature=0.0):
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature}
    }
    response = requests.post(url, json=payload, timeout=40)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"Gemini API Error: {response.text}")

def call_free_cloud_gateway(prompt, temperature=0.0):
    """
    Automatic Zero-Config Free Cloud LLM Gateway (Pollinations API).
    100% Free, Zero Key, Unlimited, High Performance.
    """
    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "openai",
        "temperature": temperature,
        "seed": 42
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MultiAgent-RAG-System/1.0"
    }
    
    # Attempt with retries
    for model_name in ["openai", "mistral", "searchgpt"]:
        try:
            payload["model"] = model_name
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text.strip()
        except Exception:
            continue
            
    raise RuntimeError("Free cloud gateway momentarily busy.")

def call_local_extractive_fallback(prompt):
    """
    Pure Python Extractive Grounding Fallback:
    Extracts key sentences and facts from the prompt context directly.
    """
    # Look for CONTEXT or DOCUMENT in prompt
    context_match = re.search(r"(?:RETRIEVED CONTEXT|DOCUMENT CONTEXT|DOCUMENT EVIDENCE)[:\s]+([\s\S]+?)(?=\n[A-Z\s]{4,}:|$)", prompt, re.IGNORECASE)
    context_text = context_match.group(1).strip() if context_match else prompt
    
    lines = [l.strip() for l in context_text.split("\n") if l.strip() and not l.startswith("---") and not l.startswith("===")]
    summary_points = lines[:8] if len(lines) >= 8 else lines
    
    return "Based on the retrieved document context:\n\n" + "\n".join([f"• {p}" for p in summary_points if len(p) > 10])

def call_llm(prompt, model="llama3.2:latest", temperature=0.0):
    """
    Universal 5-Tier Fail-Safe LLM Dispatcher:
    1. User's Groq Key (Fastest if provided)
    2. User's Gemini / OpenAI Key (if provided)
    3. Local Ollama (if accessible)
    4. Free Zero-Config Cloud Inference Gateway (Works on Streamlit Cloud without keys!)
    5. Pure Python Extractive Grounding Fallback (Never crashes)
    """
    groq_key = get_secret("GROQ_API_KEY")
    if groq_key:
        try:
            return call_groq(prompt, groq_key, temperature)
        except Exception as e:
            print(f"[Groq Error]: {e}")

    gemini_key = get_secret("GEMINI_API_KEY")
    if gemini_key:
        try:
            return call_gemini(prompt, gemini_key, temperature)
        except Exception as e:
            print(f"[Gemini Error]: {e}")

    openai_key = get_secret("OPENAI_API_KEY")
    if openai_key:
        try:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[OpenAI Error]: {e}")

    # 3. Try Local Ollama
    try:
        return call_ollama(prompt, model=model, temperature=temperature)
    except Exception:
        pass

    # 4. Try Free Zero-Config Cloud Gateway
    try:
        return call_free_cloud_gateway(prompt, temperature=temperature)
    except Exception as e:
        print(f"[Free Cloud Gateway Error]: {e}")

    # 5. Local Document Extractive Fallback
    return call_local_extractive_fallback(prompt)
