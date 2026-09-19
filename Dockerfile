FROM python:3.11-slim

# Install system utilities, curl, ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up non-root user (Standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . .

# Expose Hugging Face default port
EXPOSE 7860

# Switch to non-root user
USER user

# Start Ollama background service, pull model, and run Streamlit on port 7860
CMD ["sh", "-c", "ollama serve & sleep 3 && ollama pull llama3.2:latest && streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true"]
