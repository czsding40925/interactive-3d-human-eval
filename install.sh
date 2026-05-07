#!/bin/bash
# install.sh — one-time setup on the L4 server
# Run as a user with sudo access

set -e

echo "==> Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    xvfb \
    x11-utils \
    xdotool \
    ffmpeg \
    python3-pip \
    wget \
    unzip

echo "==> Installing Python dependencies..."
pip3 install websockets --break-system-packages

echo "==> Downloading mediamtx (WebRTC/RTSP server)..."
# Check latest release at https://github.com/bluenviron/mediamtx/releases
MEDIAMTX_VERSION="1.9.1"
wget -q "https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VERSION}/mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz" -O /tmp/mediamtx.tar.gz
mkdir -p ~/mediamtx
tar -xzf /tmp/mediamtx.tar.gz -C ~/mediamtx
echo "mediamtx extracted to ~/mediamtx"

echo ""
echo "==> Done. Next steps:"
echo "  1. Edit interactive-3d-human-eval/mediamtx.yml if needed"
echo "  2. Run start_noninteractive.sh or start_interactive.sh"
