#!/bin/bash
# Mycelium Universal Installer
# Run: curl -L https://raw.githubusercontent.com/HautlyS/mycelium/main/install.sh | bash
#
# Strategy:
#   1. Try downloading pre-built binary from GitHub Releases
#   2. Fall back to building from source via cargo
#   3. Link the binary into ~/.mycelium/bin
#   4. Update PATH in the user's shell config

set -euo pipefail

MYCELIUM_VERSION="v0.2.0"
MYCELIUM_REPO="HautlyS/mycelium"
BINARY_NAME="mycelium"

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}$*${NC}"; }
log_ok()    { echo -e "${GREEN}$*${NC}"; }
log_warn()  { echo -e "${YELLOW}$*${NC}"; }
log_err()   { echo -e "${RED}$*${NC}"; }

# ── Header ──────────────────────────────────────────────────────────────
echo ""
echo "🍄 Mycelium ${MYCELIUM_VERSION} Installer"
echo "=================================="
echo ""

# ── Detect OS / Arch ────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)     OS_NAME="linux" ;;
    Darwin*)    OS_NAME="darwin" ;;
    FreeBSD*)   OS_NAME="freebsd" ;;
    MINGW*|MSYS*|CYGWIN*) OS_NAME="windows" ;;
    *)
        log_err "Unsupported OS: $OS"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64)        ARCH_NAME="x86_64" ;;
    aarch64|arm64) ARCH_NAME="aarch64" ;;
    *)
        log_err "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Build the Rust target triple
if [ "$OS_NAME" = "darwin" ]; then
    TARGET="${ARCH_NAME}-apple-darwin"
elif [ "$OS_NAME" = "windows" ]; then
    TARGET="${ARCH_NAME}-pc-windows-msvc"
elif [ "$OS_NAME" = "freebsd" ]; then
    TARGET="${ARCH_NAME}-unknown-freebsd"
else
    TARGET="${ARCH_NAME}-unknown-linux-gnu"
fi

log_info "Platform: $OS_NAME ($ARCH) → target: $TARGET"

# ── Install directory ───────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.mycelium"
BIN_DIR="${INSTALL_DIR}/bin"
mkdir -p "$BIN_DIR"

log_info "Installing to: ${BIN_DIR}"

# ── Helper: try downloading a pre-built binary ──────────────────────────
download_binary() {
    local base_url="https://github.com/${MYCELIUM_REPO}/releases/latest/download"
    local candidates=(
        "${BINARY_NAME}-${TARGET}"
        "${BINARY_NAME}-${OS_NAME}-${ARCH_NAME}"
        "${BINARY_NAME}-${TARGET}.gz"
        "${BINARY_NAME}"
    )

    for name in "${candidates[@]}"; do
        local url="${base_url}/${name}"
        if curl -sfI "$url" >/dev/null 2>&1; then
            log_info "Downloading: $url"
            local dest="${BIN_DIR}/${BINARY_NAME}"

            if [[ "$name" == *.gz ]]; then
                curl -sL "$url" | gunzip - > "$dest"
            else
                curl -sL "$url" -o "$dest"
            fi

            # Sanity check: must be a non-empty file
            if [ -s "$dest" ]; then
                chmod +x "$dest"
                # Quick check that it's not an HTML error page
                local header
                header=$(head -c 6 "$dest" 2>/dev/null || true)
                if [[ "$header" != "<!"* ]] && [[ "$header" != "Not Fo"* ]] && [[ "$header" != "404" ]]; then
                    log_ok "Downloaded pre-built binary successfully"
                    return 0
                else
                    log_warn "Downloaded file is not a valid binary, removing"
                    rm -f "$dest"
                fi
            fi
        fi
    done

    return 1
}

# ── Helper: ensure Rust toolchain ───────────────────────────────────────
ensure_rust() {
    if command -v cargo &>/dev/null; then
        # Edition 2024 requires Rust 1.85+
        local rust_version
        rust_version=$(rustc --version | grep -oP '\d+\.\d+' | head -1)
        local major minor
        major=$(echo "$rust_version" | cut -d. -f1)
        minor=$(echo "$rust_version" | cut -d. -f2)

        if [ "$major" -lt 1 ] || { [ "$major" -eq 1 ] && [ "$minor" -lt 85 ]; }; then
            log_warn "Rust $rust_version detected; edition 2024 requires Rust 1.85+"
            log_info "Updating Rust toolchain..."
            rustup update stable 2>/dev/null || true
        fi
        return 0
    fi

    log_info "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    . "${HOME}/.cargo/env"

    if ! command -v cargo &>/dev/null; then
        log_err "Rust installation failed. Please install manually from https://rustup.rs"
        exit 1
    fi
}

# ── Helper: build from source ───────────────────────────────────────────
build_from_source() {
    log_info "Building Mycelium from source (this may take a while)..."

    ensure_rust

    local clone_dir
    clone_dir=$(mktemp -d)
    trap "rm -rf '$clone_dir'" EXIT

    log_info "Cloning repository..."
    git clone --depth 1 "https://github.com/${MYCELIUM_REPO}.git" "$clone_dir"

    cd "$clone_dir"

    log_info "Running cargo build --release..."
    cargo build --release --package mycelium-node 2>&1

    local built_binary="target/release/${BINARY_NAME}"
    if [ ! -f "$built_binary" ]; then
        log_err "Build completed but binary not found at $built_binary"
        exit 1
    fi

    cp "$built_binary" "${BIN_DIR}/${BINARY_NAME}"
    chmod +x "${BIN_DIR}/${BINARY_NAME}"
    log_ok "Build complete!"
}

# ── Main: install logic ─────────────────────────────────────────────────
if download_binary; then
    : # binary downloaded successfully
else
    log_warn "No pre-built binary available for $TARGET"
    echo ""

    # Interactive prompt if stdin is a terminal
    if [ -t 0 ]; then
        read -rp "Build from source? This requires Rust and may take several minutes. [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log_info "Aborted. You can build manually with:"
            echo "   cargo install --git https://github.com/${MYCELIUM_REPO}.git mycelium-node"
            exit 0
        fi
    fi

    build_from_source
fi

# ── Verify installation ─────────────────────────────────────────────────
INSTALLED_BIN="${BIN_DIR}/${BINARY_NAME}"
if [ ! -x "$INSTALLED_BIN" ]; then
    log_err "Installation failed — binary not found at ${INSTALLED_BIN}"
    exit 1
fi

log_ok "Installed: $("$INSTALLED_BIN" --version 2>/dev/null || echo "${BINARY_NAME} ${MYCELIUM_VERSION}")"

# ── Update PATH in shell config ─────────────────────────────────────────
update_shell_path() {
    local rc_file="$1"
    local export_line="export PATH=\"${BIN_DIR}:\$PATH\""

    if [ ! -f "$rc_file" ]; then
        return 1
    fi

    if grep -qF "$BIN_DIR" "$rc_file" 2>/dev/null; then
        return 0  # already present
    fi

    echo "" >> "$rc_file"
    echo "# Mycelium AI — added by installer" >> "$rc_file"
    echo "$export_line" >> "$rc_file"
    log_info "Added ${BIN_DIR} to PATH in $(basename "$rc_file")"
    return 0
}

# Update the appropriate shell config (zsh preferred, then bash)
SHELL_UPDATED=false
for rc in "${HOME}/.zshrc" "${HOME}/.bashrc" "${HOME}/.bash_profile" "${HOME}/.profile"; do
    if update_shell_path "$rc"; then
        SHELL_UPDATED=true
        break
    fi
done

if [ "$SHELL_UPDATED" = false ]; then
    log_warn "Could not find a shell config file. Add this to your PATH manually:"
    echo "   export PATH=\"${BIN_DIR}:\$PATH\""
fi

# ── Done ────────────────────────────────────────────────────────────────
echo ""
log_ok "Installation complete!"
echo ""
echo "To start Mycelium:"
echo "   ${INSTALLED_BIN} --listen 0.0.0.0:4001"
echo ""
echo "Options:"
echo "   --model /path/to/model.gguf    Load a GGUF model"
echo "   --bootstrap /ip4/...           Connect to a specific peer"
echo "   --spore-mode                   Enable auto-replication"
echo ""
echo "Web UI: https://hautlys.github.io/mycelium/"
echo "API:    http://localhost:8080"
echo ""