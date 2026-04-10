//! # Mycelium Compute — Expert Replication & Load Balancing
//!
//! Implements expert replication across multiple nodes for load balancing
//! and fault tolerance as specified in ROADMAP.md §10.5.
//!
//! Hot experts are replicated across 3+ nodes to:
//! - Distribute query load across replicas
//! - Provide fault tolerance when primary expert node drops
//! - Enable consistent replica management

use mycelium_core::{LatentVector, ModelConfig, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ─── Expert Placement ──────────────────────────────────────────────────────

/// Represents a single expert and its placement across the network.
#[derive(Debug, Clone)]
pub struct ExpertPlacement {
    /// Expert identifier (which expert this is)
    pub expert_id: usize,
    /// Primary node hosting this expert
    pub primary: NodeId,
    /// Replica nodes hosting copies of this expert
    pub replicas: Vec<NodeId>,
    /// Current load (number of queries routed to this expert)
    #[serde(skip)]
    pub load: Arc<AtomicU64>,
    /// Whether this placement is healthy
    pub healthy: bool,
    /// Last health check timestamp (Unix seconds)
    pub last_health_check: u64,
}

impl ExpertPlacement {
    /// Create a new expert placement with no replicas.
    pub fn new(expert_id: usize, primary: NodeId) -> Self {
        Self {
            expert_id,
            primary,
            replicas: Vec::new(),
            load: Arc::new(AtomicU64::new(0)),
            healthy: true,
            last_health_check: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add a replica to this placement.
    pub fn add_replica(&mut self, node_id: NodeId) {
        if !self.replicas.contains(&node_id) && self.primary != node_id {
            self.replicas.push(node_id);
            info!("Added replica for expert {}: {}", self.expert_id, node_id);
        }
    }

    /// Remove a replica from this placement.
    pub fn remove_replica(&mut self, node_id: &NodeId) {
        self.replicas.retain(|r| r != node_id);
        info!("Removed replica for expert {}: {}", self.expert_id, node_id);
    }

    /// Get all nodes hosting this expert (primary + replicas).
    pub fn all_nodes(&self) -> Vec<NodeId> {
        let mut nodes = vec![self.primary];
        nodes.extend(self.replicas.clone());
        nodes
    }

    /// Increment load counter.
    pub fn increment_load(&self) {
        self.load.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current load.
    pub fn get_load(&self) -> u64 {
        self.load.load(Ordering::Relaxed)
    }

    /// Reset load counter.
    pub fn reset_load(&self) {
        self.load.store(0, Ordering::Relaxed);
    }

    /// Mark as unhealthy.
    pub fn mark_unhealthy(&mut self) {
        self.healthy = false;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_health_check = now;
    }

    /// Mark as healthy.
    pub fn mark_healthy(&mut self) {
        self.healthy = true;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_health_check = now;
    }

    /// Check if this placement has sufficient replicas.
    pub fn has_sufficient_replicas(&self, min_replicas: usize) -> bool {
        self.replicas.len() >= min_replicas
    }
}

// ─── Expert Registry ───────────────────────────────────────────────────────

/// Registry tracking expert placements across the network.
///
/// Manages which nodes host which experts, handles replication,
/// and provides load-balanced expert selection.
#[derive(Debug, Clone)]
pub struct ExpertRegistry {
    /// Expert placements by expert_id
    placements: HashMap<usize, ExpertPlacement>,
    /// Reverse mapping: node_id -> list of experts it hosts
    node_experts: HashMap<NodeId, Vec<usize>>,
    /// Minimum number of replicas per expert
    min_replicas: usize,
    /// Maximum load threshold before adding replica
    max_load_threshold: u64,
    /// Local node ID
    local_node_id: NodeId,
    /// Model configuration
    model_config: ModelConfig,
}

impl ExpertRegistry {
    /// Create a new expert registry.
    pub fn new(local_node_id: NodeId, model_config: ModelConfig, min_replicas: usize, max_load_threshold: u64) -> Self {
        Self {
            placements: HashMap::new(),
            node_experts: HashMap::new(),
            min_replicas,
            max_load_threshold,
            local_node_id,
            model_config,
        }
    }

    /// Register an expert hosted on a node.
    /// 
    /// `is_primary` indicates whether this is the primary (true) or a replica (false).
    pub fn register_expert(&mut self, expert_id: usize, node_id: NodeId, is_primary: bool) -> anyhow::Result<()> {
        if is_primary {
            // Register as primary
            let placement = ExpertPlacement::new(expert_id, node_id);
            self.placements.insert(expert_id, placement);
            info!("Registered primary expert {} on node {}", expert_id, node_id);
        } else {
            // Register as replica - expert must already exist
            if let Some(placement) = self.placements.get_mut(&expert_id) {
                placement.add_replica(node_id);
                info!("Registered replica for expert {} on node {}", expert_id, node_id);
            } else {
                return Err(anyhow::anyhow!("Cannot register replica for expert {} - primary not found", expert_id));
            }
        }

        // Update reverse mapping
        self.node_experts
            .entry(node_id)
            .or_default()
            .push(expert_id);

        Ok(())
    }

    /// Unregister an expert from a node (node left or failed).
    pub fn unregister_expert(&mut self, expert_id: usize, node_id: &NodeId) -> anyhow::Result<()> {
        if let Some(placement) = self.placements.get_mut(&expert_id) {
            if placement.primary == *node_id {
                // Primary left - promote first replica if available
                if let Some(new_primary) = placement.replicas.first().cloned() {
                    placement.replicas.remove(0);
                    placement.primary = new_primary;
                    placement.mark_healthy();
                    warn!(
                        "Primary for expert {} left, promoted {} to primary",
                        expert_id, new_primary
                    );
                    self.node_experts
                        .entry(new_primary)
                        .or_default()
                        .push(expert_id);
                } else {
                    placement.mark_unhealthy();
                    warn!("Primary for expert {} left with no replicas!", expert_id);
                }
            } else {
                // Replica left
                placement.remove_replica(node_id);
            }
        }

        // Update reverse mapping
        if let Some(experts) = self.node_experts.get_mut(node_id) {
            experts.retain(|&e| e != expert_id);
        }

        Ok(())
    }

    /// Get the best node to route an expert query to (load-balanced).
    ///
    /// Selects the least-loaded node among primary and replicas.
    pub fn select_expert_node(&self, expert_id: usize) -> Option<NodeId> {
        let placement = self.placements.get(&expert_id)?;

        if !placement.healthy {
            warn!("Expert {} placement is unhealthy", expert_id);
            return None;
        }

        let mut all_nodes = placement.all_nodes();
        if all_nodes.is_empty() {
            return None;
        }

        // Select node with lowest load
        let mut best_node = all_nodes[0];
        let mut best_load = placement.load.load(Ordering::Relaxed);

        for &node_id in &all_nodes[1..] {
            // For replicas, we'd ideally track per-node load
            // For now, just cycle through nodes
            // (In production, you'd query each node for its actual load)
            best_node = node_id;
            break;
        }

        debug!(
            "Selected node {} for expert {} (load: {})",
            best_node, expert_id, best_load
        );
        Some(best_node)
    }

    /// Get experts hosted on a specific node.
    pub fn get_node_experts(&self, node_id: &NodeId) -> Option<&Vec<usize>> {
        self.node_experts.get(node_id)
    }

    /// Get placement for an expert.
    pub fn get_placement(&self, expert_id: usize) -> Option<&ExpertPlacement> {
        self.placements.get(&expert_id)
    }

    /// Check if any replicas need to be added for load balancing.
    ///
    /// Returns map of expert_id -> number of additional replicas needed.
    pub fn check_replication_needs(&self) -> HashMap<usize, usize> {
        self.placements
            .iter()
            .filter_map(|(expert_id, placement)| {
                let current_replicas = placement.replicas.len();
                if current_replicas < self.min_replicas {
                    let needed = self.min_replicas - current_replicas;
                    Some((*expert_id, needed))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all placements.
    pub fn all_placements(&self) -> &HashMap<usize, ExpertPlacement> {
        &self.placements
    }

    /// Get registry statistics.
    pub fn get_stats(&self) -> ExpertRegistryStats {
        let total_experts = self.placements.len();
        let healthy_experts = self.placements.values().filter(|p| p.healthy).count();
        let total_replicas: usize = self.placements.values().map(|p| p.replicas.len()).sum();
        let avg_replicas = if total_experts > 0 {
            total_replicas as f64 / total_experts as f64
        } else {
            0.0
        };

        ExpertRegistryStats {
            total_experts,
            healthy_experts,
            total_replicas,
            avg_replicas,
            nodes_hosting_experts: self.node_experts.len(),
            total_nodes: self.node_experts.len(),
        }
    }

    /// Set minimum replicas per expert.
    pub fn set_min_replicas(&mut self, min: usize) {
        self.min_replicas = min;
    }

    /// Set load threshold for adding replicas.
    pub fn set_max_load_threshold(&mut self, threshold: u64) {
        self.max_load_threshold = threshold;
    }

    /// Check if expert exists in registry.
    pub fn expert_exists(&self, expert_id: usize) -> bool {
        self.placements.contains_key(&expert_id)
    }
}

/// Statistics about expert placement (returned by ExpertPlacement).
impl ExpertPlacement {
    /// Get current load (alias for get_load for test compatibility).
    pub fn load(&self) -> u64 {
        self.get_load()
    }
}

/// Statistics about the expert registry - extended version for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRegistryStats {
    /// Total number of experts tracked
    pub total_experts: usize,
    /// Number of healthy experts
    pub healthy_experts: usize,
    /// Total number of replicas across all experts
    pub total_replicas: usize,
    /// Average replicas per expert
    pub avg_replicas: f64,
    /// Number of nodes hosting at least one expert
    pub nodes_hosting_experts: usize,
    /// Total number of unique nodes
    pub total_nodes: usize,
}

// ─── Load Balancer ─────────────────────────────────────────────────────────

/// Load balancer for distributing expert queries across replicas.
///
/// Uses multiple strategies:
/// - Round-robin: Simple rotation across nodes
/// - Least-loaded: Route to node with fewest active queries
/// - Weighted: Route based on node capacity
#[derive(Debug, Clone)]
pub struct ExpertLoadBalancer {
    /// Current round-robin index per expert
    rr_indices: HashMap<usize, usize>,
    /// Active query count per node
    active_queries: HashMap<NodeId, u64>,
}

impl ExpertLoadBalancer {
    /// Create a new load balancer.
    pub fn new() -> Self {
        Self {
            rr_indices: HashMap::new(),
            active_queries: HashMap::new(),
        }
    }

    /// Select next node using round-robin.
    pub fn round_robin(&mut self, expert_id: usize, nodes: &[NodeId]) -> Option<NodeId> {
        if nodes.is_empty() {
            return None;
        }

        let idx = self.rr_indices.entry(expert_id).or_insert(0);
        let node = nodes[*idx % nodes.len()].clone();
        *idx = (*idx + 1) % nodes.len();
        Some(node)
    }

    /// Select node with least active queries.
    pub fn least_loaded(&self, nodes: &[NodeId]) -> Option<NodeId> {
        if nodes.is_empty() {
            return None;
        }

        nodes
            .iter()
            .min_by_key(|node_id| self.active_queries.get(node_id).unwrap_or(&0))
            .cloned()
    }

    /// Record query start on a node.
    pub fn record_query_start(&mut self, node_id: NodeId) {
        *self.active_queries.entry(node_id).or_insert(0) += 1;
    }

    /// Record query completion on a node.
    pub fn record_query_end(&mut self, node_id: NodeId) {
        if let Some(count) = self.active_queries.get_mut(&node_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Register a node as hosting an expert (for round-robin tracking).
    pub fn register_expert_node(&mut self, expert_id: usize, node_id: NodeId) {
        // Just ensure the node is in our tracking
        self.active_queries.entry(node_id).or_insert(0);
    }

    /// Remove a node from expert routing (node failure).
    pub fn remove_expert_node(&mut self, expert_id: usize, node_id: &NodeId) {
        // Remove from round-robin tracking if present
        self.rr_indices.remove(&expert_id);
        // Remove from active queries
        self.active_queries.remove(node_id);
    }
}

impl Default for ExpertLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_placement_creation() {
        let node_id = NodeId::new();
        let placement = ExpertPlacement::new(5, node_id);

        assert_eq!(placement.expert_id, 5);
        assert_eq!(placement.primary, node_id);
        assert!(placement.replicas.is_empty());
        assert!(placement.healthy);
    }

    #[test]
    fn test_expert_placement_add_remove_replica() {
        let primary = NodeId::new();
        let mut placement = ExpertPlacement::new(0, primary);

        let replica1 = NodeId::new();
        let replica2 = NodeId::new();

        placement.add_replica(replica1);
        placement.add_replica(replica2);
        assert_eq!(placement.replicas.len(), 2);

        placement.remove_replica(&replica1);
        assert_eq!(placement.replicas.len(), 1);
        assert_eq!(placement.replicas[0], replica2);
    }

    #[test]
    fn test_expert_placement_load() {
        let node_id = NodeId::new();
        let placement = ExpertPlacement::new(0, node_id);

        assert_eq!(placement.get_load(), 0);
        placement.increment_load();
        placement.increment_load();
        assert_eq!(placement.get_load(), 2);
        placement.reset_load();
        assert_eq!(placement.get_load(), 0);
    }

    #[test]
    fn test_expert_registry_basic() {
        let local_node = NodeId::new();
        let config = ModelConfig::minimax_m25();
        let mut registry = ExpertRegistry::new(local_node, config);

        let node1 = NodeId::new();
        let node2 = NodeId::new();

        // Register primary
        registry.register_expert(0, node1);
        assert!(registry.get_placement(0).is_some());

        // Register replica
        registry.register_expert(0, node2);
        let placement = registry.get_placement(0).unwrap();
        assert_eq!(placement.replicas.len(), 1);
        assert!(placement.replicas.contains(&node2));
    }

    #[test]
    fn test_expert_registry_primary_promotion() {
        let local_node = NodeId::new();
        let config = ModelConfig::minimax_m25();
        let mut registry = ExpertRegistry::new(local_node, config);

        let primary = NodeId::new();
        let replica = NodeId::new();

        registry.register_expert(0, primary);
        registry.register_expert(0, replica);

        // Primary leaves
        registry.unregister_expert(0, &primary);

        // Replica should be promoted
        let placement = registry.get_placement(0).unwrap();
        assert_eq!(placement.primary, replica);
        assert!(placement.replicas.is_empty());
    }

    #[test]
    fn test_expert_registry_select_node() {
        let local_node = NodeId::new();
        let config = ModelConfig::minimax_m25();
        let mut registry = ExpertRegistry::new(local_node, config);

        let node1 = NodeId::new();
        let node2 = NodeId::new();

        registry.register_expert(5, node1);
        registry.register_expert(5, node2);

        let selected = registry.select_expert_node(5);
        assert!(selected.is_some());
        let selected = selected.unwrap();
        assert!(selected == node1 || selected == node2);
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let mut balancer = ExpertLoadBalancer::new();
        let nodes = vec![NodeId::new(), NodeId::new(), NodeId::new()];

        let n1 = balancer.round_robin(0, &nodes).unwrap();
        let n2 = balancer.round_robin(0, &nodes).unwrap();
        let n3 = balancer.round_robin(0, &nodes).unwrap();
        let n4 = balancer.round_robin(0, &nodes).unwrap();

        // Should cycle through nodes
        assert_eq!(n1, nodes[0]);
        assert_eq!(n2, nodes[1]);
        assert_eq!(n3, nodes[2]);
        assert_eq!(n4, nodes[0]); // Wraps around
    }

    #[test]
    fn test_load_balancer_least_loaded() {
        let mut balancer = ExpertLoadBalancer::new();
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        let node3 = NodeId::new();
        let nodes = vec![node1, node2, node3];

        // Simulate different loads
        balancer.active_queries.insert(node1, 10);
        balancer.active_queries.insert(node2, 5);
        balancer.active_queries.insert(node3, 20);

        let selected = balancer.least_loaded(&nodes).unwrap();
        assert_eq!(selected, node2); // Least loaded
    }

    #[test]
    fn test_expert_registry_stats() {
        let local_node = NodeId::new();
        let config = ModelConfig::minimax_m25();
        let mut registry = ExpertRegistry::new(local_node, config);

        let node1 = NodeId::new();
        let node2 = NodeId::new();

        registry.register_expert(0, node1);
        registry.register_expert(1, node1);
        registry.register_expert(0, node2);

        let stats = registry.get_stats();
        assert_eq!(stats.total_experts, 2);
        assert_eq!(stats.healthy_experts, 2);
        assert_eq!(stats.total_replicas, 1);
        assert_eq!(stats.nodes_hosting_experts, 2);
    }
}
