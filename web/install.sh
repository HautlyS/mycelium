#!/bin/bash
# Mycelium One-Line Installer
# Run: curl -L mycelium.ai/install | bash
# Or:  curl -L https://raw.githubusercontent.com/HautlyS/mycelium/main/install.sh | bash

set -e

MYCELIUM_VERSION="v0.2.0"
MYCELIUM_REPO="HautlyS/mycelium"

echo "🍄 Installing Mycelium $MYCELIUM_VERSION..."

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)     PLATFORM="unknown-linux-gnu" ;;
    Darwin*)   PLATFORM="apple-darwin" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="pc-windows-msvc" ;;
    *)         PLATFORM="unknown-linux-gnu" ;;
esac

case "$ARCH" in
    x86_64)      BINARY="x86_64" ;;
    aarch64|arm64) BINARY="aarch64" ;;
    arm*)        BINARY="arm" ;;
    *)          BINARY="x86_64" ;;
esac

# Download release
TARBALL="mycelium-${PLATFORM}.tar.gz"
DOWNLOAD_URL="https://github.com/${MYCELIUM_REPO}/releases/latest/download/${TARBALL}"
INSTALL_DIR="${HOME}/.mycelium"

# Create install directory
mkdir -p "$INSTALL_DIR/bin"

# Download using curl or wget
echo "📥 Downloading from GitHub..."
if command -v curl &> /dev/null; then
    curl -sL "$DOWNLOAD_URL" -o "/tmp/${TARBALL}" || {
        echo "❌ Download failed. Trying alternative..."
        # Fallback: download native binary directly
        NATIVE_URL="https://github.com/${MYCELIUM_REPO}/releases/latest/download/mycelium-node"
        curl -sL "$NATIVE_URL" -o "$INSTALL_DIR/bin/mycelium-node"
        chmod +x "$INSTALL_DIR/bin/mycelium-node"
        echo "✅ Installed to $INSTALL_DIR/bin/mycelium-node"
        exit 0
    }
elif command -v wget &> /dev/null; then
    wget -q "$DOWNLOAD_URL" -O "/tmp/${TARBALL}" || {
        echo "❌ Download failed"
        exit 1
    }
else
    echo "❌ curl or wget required"
    exit 1
fi

# Extract
echo "📦 Extracting..."
tar -xzf "/tmp/${TARBALL}" -C "$INSTALL_DIR/bin/" 2>/dev/null || {
    # If tar fails, try as plain binary
    mv "/tmp/${TARBALL}" "$INSTALL_DIR/bin/mycelium-node" 2>/dev/null || true
}

chmod +x "$INSTALL_DIR/bin/mycelium-node"

# Add to PATH
SHELL_RC="${HOME}/.bashrc"
if [ -f "${HOME}/.zshrc" ]; then SHELL_RC="${HOME}/.zshrc"; fi

if ! grep -q "mycelium" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Mycelium" >> "$SHELL_RC"
    echo "export PATH=\"\$PATH:$INSTALL_DIR/bin\"" >> "$SHELL_RC"
fi

echo "✅ Mycelium installed to $INSTALL_DIR/bin/mycelium-node"
echo ""
echo "🍄 To start:"
echo "   mycelium-node --listen 0.0.0.0:4001"
echo ""
echo "🌐 Web interface: https://HautlyS.github.io/mycelium"
echo ""