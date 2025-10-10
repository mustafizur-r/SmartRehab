#!/usr/bin/env bash
set -e

echo "──────────────────────────────────────────────"
echo "     Universal Ollama Installation Script"
echo "──────────────────────────────────────────────"

OS=$(uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]')

# ---------- Linux / Docker / WSL ----------
if [[ "$OS" == "linux" ]]; then
  echo "[+] Installing on Linux..."
  curl -fsSL https://ollama.com/install.sh | sh
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 5
  ollama pull llama3
  ollama pull qwen2.5:7b-instruct-q4_K_M

# ---------- macOS ----------
elif [[ "$OS" == "darwin" ]]; then
  echo "[+] Installing on macOS..."
  curl -fsSL https://ollama.com/download/Ollama-darwin.zip -o /tmp/Ollama.zip
  unzip -q /tmp/Ollama.zip -d /Applications/
  /Applications/Ollama.app/Contents/MacOS/Ollama serve >/tmp/ollama.log 2>&1 &
  sleep 5
  ollama pull llama3
  ollama pull qwen2.5:7b-instruct-q4_K_M

# ---------- Windows ----------
else
  echo "[+] Detected Windows environment."
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoLogo -ExecutionPolicy Bypass -Command "
      Write-Host 'Installing Ollama for Windows...';
      \$installer = [IO.Path]::Combine([IO.Path]::GetTempPath(),'OllamaSetup.exe');
      Invoke-WebRequest 'https://ollama.com/download/OllamaSetup.exe' -OutFile \$installer;
      Start-Process \$installer -Wait;
      Write-Host 'Starting Ollama service...';
      Start-Process 'ollama' -ArgumentList 'serve' -WindowStyle Hidden;
      Start-Sleep -Seconds 5;
      ollama pull llama3;
      ollama pull qwen2.5:7b-instruct-q4_K_M;
    "
  else
    echo "⚠️  PowerShell not found. Please install manually from https://ollama.com/download"
    exit 1
  fi
fi

echo "──────────────────────────────────────────────"
echo "✅ Ollama + Llama3 installed successfully!"
echo "──────────────────────────────────────────────"
