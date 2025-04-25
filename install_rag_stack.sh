#!/bin/bash

echo "🛠️ Installing the basic RAG stack..."

# Update the system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential curl git unzip htop python3 python3-venv python3-pip docker.io docker-compose tesseract-ocr cifs-utils

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Install Ollama (local LLM engine)
curl -fsSL https://ollama.com/install.sh | sh

# Open port 11434 for Ollama if UFW firewall is active
if sudo ufw status | grep -q "Status: active"; then
  sudo ufw allow 11434/tcp
fi

# Create the project directory
mkdir -p ~/rag-stack/rag_data
cd ~/rag-stack

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ RAG stack installation complete! Ready to use."
