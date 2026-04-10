//! # Mycelium iOS FFI Bridge
//!
//! Rust FFI entry points for iOS BGTaskScheduler integration.
//! These functions are called from Swift to perform P2P sync
//! during background execution windows.
//!
//! See CROSSDEVICE.md §8.6.4 for the Swift-side bridge code.

use std::sync::atomic::{AtomicBool, Ordering};

/// Global cancellation flag for background tasks.
/// Set to `true` by `mycelium_cancel_all_operations()` when iOS
/// is about to terminate our background time.
static CANCEL_FLAG: AtomicBool = AtomicBool::new(false);

/// Called from BGTaskScheduler — runs libp2p DHT refresh + spore sync.
///
/// This function creates a single-threaded tokio runtime and runs the
/// sync operations within the specified timeout. The BGProcessingTask
/// typically gives ~30 seconds; we use a self-imposed timeout of 25s
/// to leave 5s for cleanup.
///
/// Returns `true` if sync completed successfully, `false` otherwise.
///
/// # Safety
/// This function is safe to call from Swift via FFI.
#[no_mangle]
pub extern "C" fn mycelium_p2p_sync_background(_timeout_secs: u64) -> bool {
    // TODO: Implement actual P2P sync when hyphae DHT refresh and spore
    // sync APIs are available. Currently returning false to indicate
    // no sync was performed.
    tracing::warn!("mycelium_p2p_sync_background: not yet implemented");
    false
}

/// Called by expirationHandler — must cancel all work immediately.
///
/// This sets the global cancellation flag. All async operations should
/// check this flag periodically and abort when it becomes true.
///
/// # Safety
/// This function is safe to call from Swift via FFI.
#[no_mangle]
pub extern "C" fn mycelium_cancel_all_operations() {
    CANCEL_FLAG.store(true, Ordering::SeqCst);
    tracing::warn!("Background task expired — cancelling all operations");
}

/// Check if the current background task has been cancelled.
///
/// Returns `true` if `mycelium_cancel_all_operations()` has been called.
pub fn is_cancelled() -> bool {
    CANCEL_FLAG.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_flag() {
        // Reset
        CANCEL_FLAG.store(false, Ordering::SeqCst);
        assert!(!is_cancelled());

        // Set
        CANCEL_FLAG.store(true, Ordering::SeqCst);
        assert!(is_cancelled());

        // Reset
        CANCEL_FLAG.store(false, Ordering::SeqCst);
        assert!(!is_cancelled());
    }

    #[test]
    fn test_ffi_functions() {
        // Test that FFI functions can be called without crashing
        CANCEL_FLAG.store(false, Ordering::SeqCst);

        // mycelium_p2p_sync_background is not yet implemented, always returns false
        let result = mycelium_p2p_sync_background(1);
        assert!(!result, "mycelium_p2p_sync_background should return false until implemented");

        mycelium_cancel_all_operations();
        assert!(is_cancelled());
    }
}
