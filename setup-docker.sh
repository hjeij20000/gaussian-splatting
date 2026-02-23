#!/bin/bash
# =============================================================================
# Docker + NVIDIA Container Toolkit installer for the video-to-3dgs pipeline
# Run with: bash setup-docker.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo " Installing Docker Engine"
echo "============================================"

# Docker prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker apt repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker "$USER"

echo ""
echo "============================================"
echo " Installing NVIDIA Container Toolkit"
echo "============================================"

# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "============================================"
echo " Done! Verifying installation..."
echo "============================================"

docker --version
nvidia-container-toolkit --version 2>/dev/null || echo "(nvidia-container-toolkit CLI not available, but that's OK)"

echo ""
echo "NOTE: You were added to the 'docker' group."
echo "Either log out and back in, or run: newgrp docker"
echo ""
echo "Then build the image with:"
echo "  cd /home/ibrahim"
echo "  docker build -f gaussian-splatting/Dockerfile -t video-to-3dgs ."
echo ""
echo "Run with MASt3R backend (recommended):"
echo "  docker run --gpus all -v /path/to/data:/data video-to-3dgs \\"
echo "      --video /data/video.mp4 --output /data/output --sfm-backend mast3r"
