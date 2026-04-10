#!/bin/bash
# Mycelium One-Line Installer
# Run: curl -L https://hautlys.github.io/mycelium/install.sh | bash
# Or:  curl -L https://raw.githubusercontent.com/HautlyS/mycelium/main/install.sh | bash

set -e

MYCELIUM_VERSION="v0.2.0"
MYCELIUM_REPO="HautlyS/mycelium"

echo "🍄 Mycelium $MYCELIUM_VERSION Installer"
echo "=================================="

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)     PLATFORM="x86_64-unknown-linux-gnu" ;;
    Darwin*)   PLATFORM="x86_64-apple-darwin" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="x86_64-pc-windows-msvc" ;;
    *)         PLATFORM="x86_64-unknown-linux-gnu" ;;
esac

case "$ARCH" in
    x86_64)      ;;
    aarch64|arm64) PLATFORM="aarch64-unknown-linux-gnu" ;;
esac

INSTALL_DIR="${HOME}/.mycelium"
BIN_DIR="${INSTALL_DIR}/bin"

echo "Platform: $PLATFORM"
echo "Installing to: ${BIN_DIR}"

mkdir -p "$BIN_DIR"

# Download the latest release
BASE_URL="https://github.com/${MYCELIUM_REPO}/releases/latest/download"

# Download binary
echo "📥 Downloading..."
SUCCESS=false

for NAME in "mycelium-node" "mycelium-${PLATFORM}"; do
    URL="${BASE_URL}/${NAME}"
    if curl -sfI "$URL" >/dev/null 2>&1; then
        echo "Downloading: $URL"
        if curl -sL "$URL" -o "${BIN_DIR}/mycelium-node"; then
            SUCCESS=true
            break
        fi
    fi
done

if [ "$SUCCESS" = "false" ]; then
    # Try cargo install as fallback
    echo "📦 Trying cargo install..."
    if command -v cargo &> /dev/null; then
        cargo install --git "https://github.com/${MYCELIUM_REPO}.git" --locked || true
    else
        echo "❌ Download failed. Install Rust from https://rustup.rs and run:"
        echo "   cargo install --git https://github.com/${MYCELIUM_REPO}.git"
        exit 1
    fi
fi

chmod +x "${BIN_DIR}/mycelium-node" 2>/dev/null || true

# Add to PATH
SHELL_RC="${HOME}/.bashrc"
[ -f "${HOME}/.zshrc" ] && SHELL_RC="${HOME}/.zshrc"

if ! grep -q "mycelium" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Mycelium AI" >> "$SHELL_RC"
    echo "export PATH=\"\$PATH:${BIN_DIR}\"" >> "$SHELL_RC"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run:"
echo "   ${BIN_DIR}/mycelium-node --listen 0.0.0.0:4001"
echo ""
echo "Web UI: https://hautlys.github.io/mycelium/"
echo "API:   http://localhost:8080"