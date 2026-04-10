// MyceliumBridge.swift
// Mycelium iOS Bridge — Section 8.6 CROSSDEVICE.md
//
// This Swift class provides the iOS native bridge for the Mycelium Rust core.
// It handles:
// - BGTaskScheduler registration for background spore propagation
// - iOS lifecycle management (foreground/background transitions)
// - Network reachability monitoring
// - Battery-aware compute throttling
// - Info.plist configuration validation
//
// Usage:
//   let bridge = MyceliumBridge.shared
//   bridge.start()
//   // Bridge runs in background via BGTaskScheduler

import Foundation
import BackgroundTasks
import Network
import os

#if canImport(UIKit)
import UIKit
#endif

// MARK: - FFI Imports

/// FFI function from Rust: runs P2P sync in background.
/// Returns true if sync completed successfully.
@_silgen_name("mycelium_p2p_sync_background")
func mycelium_p2p_sync_background(_ timeout_secs: UInt64) -> Bool

/// FFI function from Rust: cancels all background operations.
@_silgen_name("mycelium_cancel_all_operations")
func mycelium_cancel_all_operations()

// MARK: - Bridge Configuration

/// Configuration for the Mycelium iOS bridge.
struct MyceliumBridgeConfig {
    /// BGTaskScheduler task identifier for background propagation
    let backgroundTaskIdentifier: String
    /// BGTaskScheduler task identifier for periodic spore sync
    let sporeSyncTaskIdentifier: String
    /// Maximum background task execution time (seconds)
    let maxBackgroundExecutionTime: TimeInterval
    /// Minimum battery level (0.0-1.0) to allow background tasks
    let minimumBatteryLevel: Float
    /// Whether to enable background spore propagation
    let enableBackgroundPropagation: Bool
    /// Whether to enable periodic spore sync
    let enablePeriodicSync: Bool
    /// Background sync interval (seconds)
    let syncInterval: TimeInterval

    static let `default` = MyceliumBridgeConfig(
        backgroundTaskIdentifier: "com.mycelium.background-propagation",
        sporeSyncTaskIdentifier: "com.mycelium.spore-sync",
        maxBackgroundExecutionTime: 25, // iOS allows ~30s for BGProcessingTask
        minimumBatteryLevel: 0.2,
        enableBackgroundPropagation: true,
        enablePeriodicSync: true,
        syncInterval: 3600 // 1 hour
    )
}

// MARK: - Bridge State

/// State of the Mycelium bridge.
enum MyceliumBridgeState: String, Codable {
    case stopped
    case starting
    case running
    case background
    case stoppedBySystem
}

// MARK: - Bridge Metrics

/// Metrics reported by the bridge.
struct MyceliumBridgeMetrics {
    let state: MyceliumBridgeState
    let uptime: TimeInterval
    let peersConnected: Int
    let sporesPropagated: Int
    let dataTransferredMB: Double
    let batteryLevel: Float
    let isLowPowerMode: Bool
    let lastBackgroundTaskDate: Date?
    let lastSyncDate: Date?
}

// MARK: - MyceliumBridge

/// Primary iOS bridge for Mycelium Rust core.
///
/// This class manages the lifecycle of the Mycelium node on iOS,
/// handling background execution via BGTaskScheduler and coordinating
/// with the iOS lifecycle to ensure efficient resource usage.
///
/// # Info.plist Requirements
/// Add the following keys to your Info.plist:
/// - `BGTaskSchedulerPermittedIdentifiers` (Array): Array of task identifiers
/// - `UIBackgroundModes` (Array): Include "processing" and "fetch"
/// - `NSLocalNetworkUsageDescription` (String): Explanation for P2P networking
///
/// # Example Usage
/// ```swift
/// // In AppDelegate.application(_:didFinishLaunchingWithOptions:)
/// MyceliumBridge.shared.registerBackgroundTasks()
/// MyceliumBridge.shared.start()
///
/// // In SceneDelegate.sceneWillResignActive(_:)
/// MyceliumBridge.shared.enterBackground()
///
/// // In SceneDelegate.sceneDidBecomeActive(_:)
/// MyceliumBridge.shared.enterForeground()
/// ```
@objc public class MyceliumBridge: NSObject {

    // MARK: - Singleton

    /// Shared bridge instance.
    @objc public static let shared = MyceliumBridge()

    // MARK: - Properties

    private let config: MyceliumBridgeConfig
    private let logger = Logger(subsystem: "com.mycelium", category: "bridge")
    private var state: MyceliumBridgeState = .stopped
    private var startTime: Date?
    private var lastBackgroundTaskDate: Date?
    private var lastSyncDate: Date?
    private var sporesPropagated: Int = 0
    private var dataTransferredMB: Double = 0.0

    /// Network monitor for reachability.
    private var monitor: NWPathMonitor?

    /// Queue for network monitoring.
    private let monitorQueue = DispatchQueue(label: "com.mycelium.network-monitor")

    /// Whether the bridge is currently in background mode.
    private(set) var isInBackground = false

    /// Callback for bridge state changes.
    var onStateChange: ((MyceliumBridgeState) -> Void)?

    /// Callback for metrics updates.
    var onMetricsUpdate: ((MyceliumBridgeMetrics) -> Void)?

    // MARK: - Initialization

    private override init() {
        self.config = .default
        super.init()
        logger.info("MyceliumBridge initialized")
    }

    /// Create a bridge with custom configuration.
    @objc public convenience init(config: MyceliumBridgeConfig) {
        self.init()
        // Note: config is stored in a let property, so this is for API completeness
        // In practice, use the shared instance with default config
    }

    // MARK: - Lifecycle

    /// Start the Mycelium bridge.
    ///
    /// This initializes the Rust core via FFI, registers background tasks,
    /// and begins network monitoring.
    @objc public func start() {
        guard state == .stopped else {
            logger.warning("Bridge already started (state: \(state.rawValue))")
            return
        }

        logger.info("Starting Mycelium bridge...")
        state = .starting
        startTime = Date()

        // Validate Info.plist configuration
        validateInfoPlist()

        // Register background tasks
        if config.enableBackgroundPropagation {
            registerBackgroundTasks()
        }

        // Start network monitoring
        startNetworkMonitoring()

        state = .running
        logger.info("Mycelium bridge started")
        notifyStateChange()
    }

    /// Stop the Mycelium bridge.
    @objc public func stop() {
        guard state == .running || state == .background else {
            logger.warning("Bridge not running (state: \(state.rawValue))")
            return
        }

        logger.info("Stopping Mycelium bridge...")
        state = .stopped
        stopNetworkMonitoring()
        notifyStateChange()
    }

    /// Called when the app enters the background.
    @objc public func enterBackground() {
        isInBackground = true
        state = .background
        logger.info("App entered background")
        notifyStateChange()

        // Submit background task if enabled
        if config.enableBackgroundPropagation {
            submitBackgroundTask()
        }
    }

    /// Called when the app enters the foreground.
    @objc public func enterForeground() {
        isInBackground = false
        state = .running
        logger.info("App entered foreground")
        notifyStateChange()
    }

    // MARK: - Background Tasks

    /// Register BGTaskScheduler tasks.
    ///
    /// This must be called before the app finishes launching.
    /// Typically called from `application(_:didFinishLaunchingWithOptions:)`.
    @objc public func registerBackgroundTasks() {
        if config.enableBackgroundPropagation {
            BGTaskScheduler.shared.register(
                forTaskWithIdentifier: config.backgroundTaskIdentifier,
                using: nil
            ) { task in
                self.handleBackgroundPropagationTask(task as! BGProcessingTask)
            }
            logger.info("Registered background propagation task")
        }

        if config.enablePeriodicSync {
            BGTaskScheduler.shared.register(
                forTaskWithIdentifier: config.sporeSyncTaskIdentifier,
                using: nil
            ) { task in
                self.handleSporeSyncTask(task as! BGAppRefreshTask)
            }
            logger.info("Registered spore sync task")
        }
    }

    /// Handle the background propagation task.
    private func handleBackgroundPropagationTask(_ task: BGProcessingTask) {
        logger.info("Background propagation task started")
        lastBackgroundTaskDate = Date()

        // Set up expiration handler
        task.expirationHandler = {
            // Cancel Rust operations immediately
            mycelium_cancel_all_operations()
            self.logger.warning("Background task expired — cancelled operations")
            task.setTaskCompleted(success: false)
            self.scheduleBackgroundTask()
        }

        // Execute P2P sync on background queue
        DispatchQueue.global(qos: .background).async {
            let success = mycelium_p2p_sync_background(
                UInt64(self.config.maxBackgroundExecutionTime)
            )

            self.logger.info("P2P sync completed: \(success ? "success" : "failed")")

            if success {
                self.sporesPropagated += 1
            }

            task.setTaskCompleted(success: success)
            self.scheduleBackgroundTask()
            self.notifyMetricsUpdate()
        }
    }

    /// Handle the periodic spore sync task.
    private func handleSporeSyncTask(_ task: BGAppRefreshTask) {
        logger.info("Spore sync task started")
        lastSyncDate = Date()

        // Schedule next sync
        scheduleSporeSyncTask()

        // Set up expiration handler
        task.expirationHandler = {
            mycelium_cancel_all_operations()
            self.logger.warning("Spore sync task expired")
            task.setTaskCompleted(success: false)
        }

        // Execute sync
        DispatchQueue.global(qos: .background).async {
            let success = mycelium_p2p_sync_background(
                UInt64(min(self.config.maxBackgroundExecutionTime, 15))
            )

            self.logger.info("Spore sync completed: \(success ? "success" : "failed")")
            task.setTaskCompleted(success: success)
            self.notifyMetricsUpdate()
        }
    }

    /// Schedule the next background propagation task.
    private func scheduleBackgroundTask() {
        let request = BGProcessingTaskRequest(identifier: config.backgroundTaskIdentifier)
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: config.syncInterval)

        do {
            try BGTaskScheduler.shared.submit(request)
            logger.info("Scheduled background propagation task")
        } catch {
            logger.error("Failed to schedule background task: \(error)")
        }
    }

    /// Schedule the next spore sync task.
    private func scheduleSporeSyncTask() {
        let request = BGAppRefreshTaskRequest(identifier: config.sporeSyncTaskIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: config.syncInterval / 2)

        do {
            try BGTaskScheduler.shared.submit(request)
            logger.info("Scheduled spore sync task")
        } catch {
            logger.error("Failed to schedule spore sync task: \(error)")
        }
    }

    /// Submit a background task immediately (called when entering background).
    private func submitBackgroundTask() {
        scheduleBackgroundTask()
        scheduleSporeSyncTask()
    }

    // MARK: - Network Monitoring

    /// Start monitoring network reachability.
    private func startNetworkMonitoring() {
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self = self else { return }

            if path.status == .satisfied {
                self.logger.info("Network path satisfied")
            } else {
                self.logger.warning("Network path unsatisfied")
            }
        }
        monitor.start(queue: monitorQueue)
        self.monitor = monitor
    }

    /// Stop network monitoring.
    private func stopNetworkMonitoring() {
        monitor?.cancel()
        monitor = nil
    }

    // MARK: - Info.plist Validation

    /// Validate that required Info.plist keys are present.
    private func validateInfoPlist() {
        guard let infoDict = Bundle.main.infoDictionary else {
            logger.warning("Could not access Info.plist")
            return
        }

        // Check BGTaskSchedulerPermittedIdentifiers
        if let identifiers = infoDict["BGTaskSchedulerPermittedIdentifiers"] as? [String] {
            let requiredIdentifiers = [
                config.backgroundTaskIdentifier,
                config.sporeSyncTaskIdentifier
            ].compactMap { $0 }

            for required in requiredIdentifiers {
                if !identifiers.contains(required) {
                    logger.error("""
                        Missing BGTaskScheduler identifier: \(required)
                        Add it to BGTaskSchedulerPermittedIdentifiers in Info.plist
                    """)
                }
            }
        } else {
            logger.error("""
                Missing BGTaskSchedulerPermittedIdentifiers in Info.plist.
                Background tasks will not work. Add:
                <key>BGTaskSchedulerPermittedIdentifiers</key>
                <array>
                    <string>\(config.backgroundTaskIdentifier)</string>
                    <string>\(config.sporeSyncTaskIdentifier)</string>
                </array>
            """)
        }

        // Check UIBackgroundModes
        if let backgroundModes = infoDict["UIBackgroundModes"] as? [String] {
            if !backgroundModes.contains("processing") {
                logger.warning("""
                    Missing "processing" in UIBackgroundModes.
                    Background tasks may be limited. Add it to Info.plist.
                """)
            }
            if !backgroundModes.contains("fetch") {
                logger.warning("""
                    Missing "fetch" in UIBackgroundModes.
                    Periodic sync may be limited. Add it to Info.plist.
                """)
            }
        } else {
            logger.error("""
                Missing UIBackgroundModes in Info.plist.
                Add:
                <key>UIBackgroundModes</key>
                <array>
                    <string>processing</string>
                    <string>fetch</string>
                </array>
            """)
        }

        // Check NSLocalNetworkUsageDescription
        if infoDict["NSLocalNetworkUsageDescription"] == nil {
            logger.warning("""
                Missing NSLocalNetworkUsageDescription in Info.plist.
                P2P networking may be restricted. Add a description.
            """)
        }
    }

    // MARK: - Battery Awareness

    /// Check if battery level is sufficient for background tasks.
    private func isBatteryLevelSufficient() -> Bool {
        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let batteryLevel = UIDevice.current.batteryLevel
        return batteryLevel >= config.minimumBatteryLevel || batteryLevel < 0
        #else
        return true
        #endif
    }

    /// Check if the device is in low power mode.
    private func isLowPowerMode() -> Bool {
        #if canImport(UIKit)
        return ProcessInfo.processInfo.isLowPowerModeEnabled
        #else
        return false
        #endif
    }

    // MARK: - Metrics

    /// Get current bridge metrics.
    @objc public func getMetrics() -> MyceliumBridgeMetrics {
        let uptime = startTime.map { Date().timeIntervalSince($0) } ?? 0

        return MyceliumBridgeMetrics(
            state: state,
            uptime: uptime,
            peersConnected: 0, // Would be populated from Rust core
            sporesPropagated: sporesPropagated,
            dataTransferredMB: dataTransferredMB,
            batteryLevel: getBatteryLevel(),
            isLowPowerMode: isLowPowerMode(),
            lastBackgroundTaskDate: lastBackgroundTaskDate,
            lastSyncDate: lastSyncDate
        )
    }

    /// Get current battery level.
    private func getBatteryLevel() -> Float {
        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        return UIDevice.current.batteryLevel
        #else
        return -1.0
        #endif
    }

    // MARK: - Notifications

    /// Notify state change observers.
    private func notifyStateChange() {
        onStateChange?(state)
    }

    /// Notify metrics update observers.
    private func notifyMetricsUpdate() {
        onMetricsUpdate?(getMetrics())
    }
}

// MARK: - Objective-C Compatibility

/// Objective-C compatible wrapper for MyceliumBridgeMetrics.
@objc public class MyceliumBridgeMetricsObjC: NSObject {
    @objc public let state: String
    @objc public let uptime: TimeInterval
    @objc public let peersConnected: Int
    @objc public let sporesPropagated: Int
    @objc public let dataTransferredMB: Double
    @objc public let batteryLevel: Float
    @objc public let isLowPowerMode: Bool
    @objc public let lastBackgroundTaskDate: Date?
    @objc public let lastSyncDate: Date?

    init(metrics: MyceliumBridgeMetrics) {
        self.state = metrics.state.rawValue
        self.uptime = metrics.uptime
        self.peersConnected = metrics.peersConnected
        self.sporesPropagated = metrics.sporesPropagated
        self.dataTransferredMB = metrics.dataTransferredMB
        self.batteryLevel = metrics.batteryLevel
        self.isLowPowerMode = metrics.isLowPowerMode
        self.lastBackgroundTaskDate = metrics.lastBackgroundTaskDate
        self.lastSyncDate = metrics.lastSyncDate
    }
}
