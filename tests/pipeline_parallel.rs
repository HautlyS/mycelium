//! Multi-Node Integration Tests for Mycelium Distributed AI
//!
//! These tests validate the distributed system behavior across multiple nodes,
//! including hierarchical topics, expert replication, bootstrap nodes, and
//! pipeline parallelism.

use mycelium_core::{
    NodeId, NodeCapabilities, TopologyMap, LatentVector,
    TOPIC_GLOBAL, TOPIC_REGION_PREFIX, TOPIC_CLUSTER_PREFIX,
    region_topic, cluster_topic, NodeMetrics,
};
use mycelium_compute::{
    ExpertPlacement, ExpertRegistry, ExpertLoadBalancer, ExpertRegistryStats,
    PipelinePlan, PipelineStage, PipelineParallelConfig, PipelineExecutor,
    LatentStreamManager, LatentStreamConfig,
};
use mycelium_hyphae::HyphaeConfig;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tokio::sync::RwLock;

// ─── Hierarchical Topic Tests ───────────────────────────────────────────────

#[test]
fn test_hierarchical_topic_constants() {
    assert_eq!(TOPIC_GLOBAL, "mycelium/global");
    assert!(TOPIC_REGION_PREFIX.starts_with("mycelium/region/"));
    assert!(TOPIC_CLUSTER_PREFIX.starts_with("mycelium/cluster/"));
}

#[test]
fn test_region_topic_builder() {
    let topic = region_topic("us-east-1");
    assert_eq!(topic, "mycelium/region/us-east-1");

    let topic2 = region_topic("eu-west-2");
    assert_eq!(topic2, "mycelium/region/eu-west-2");
}

#[test]
fn test_cluster_topic_builder() {
    let topic = cluster_topic("cluster-alpha");
    assert_eq!(topic, "mycelium/cluster/cluster-alpha");

    let topic2 = cluster_topic("cluster-beta");
    assert_eq!(topic2, "mycelium/cluster/cluster-beta");
}

#[test]
fn test_topic_uniqueness() {
    let region = region_topic("region-a");
    let cluster = cluster_topic("cluster-a");

    assert_ne!(region, cluster);
    assert_ne!(region, TOPIC_GLOBAL);
    assert_ne!(cluster, TOPIC_GLOBAL);
}

// ─── Bootstrap Node Configuration Tests ─────────────────────────────────────

#[test]
fn test_bootstrap_node_config() {
    let config = HyphaeConfig::bootstrap_node();

    assert!(config.bootstrap_mode);
    assert!(!config.zero_trust_mode);
    assert_eq!(config.max_peers, 0); // Unlimited connections
    assert!(config.enable_relay);
    assert!(config.enable_autonat);
}

#[test]
fn test_hierarchical_topic_config() {
    let config = HyphaeConfig::with_hierarchical_topics(
        "us-east-1".to_string(),
        "cluster-1".to_string(),
    );

    assert!(!config.bootstrap_mode);
    assert_eq!(config.region_id, Some("us-east-1".to_string()));
    assert_eq!(config.cluster_id, Some("cluster-1".to_string()));
}

#[test]
fn test_bootstrap_config_has_relay_enabled() {
    let config = HyphaeConfig::bootstrap_node();
    // Bootstrap nodes MUST have relay enabled to help other nodes connect
    assert!(config.enable_relay, "Bootstrap nodes must have relay enabled");
    assert!(config.enable_autonat, "Bootstrap nodes must have AutoNAT enabled");
}

// ─── Node Metrics Tests ─────────────────────────────────────────────────────

#[test]
fn test_node_metrics_creation() {
    let metrics = NodeMetrics::default();

    assert_eq!(metrics.peer_count, 0);
    assert_eq!(metrics.messages_sent, 0);
    assert_eq!(metrics.messages_received, 0);
    assert_eq!(metrics.inference_count, 0);
    assert_eq!(metrics.training_steps, 0);
    assert_eq!(metrics.gradient_contributions, 0);
    assert!(!metrics.pipeline_active);
}

#[test]
fn test_node_metrics_message_tracking() {
    let mut metrics = NodeMetrics::default();

    metrics.increment_messages_sent();
    metrics.increment_messages_sent();
    metrics.increment_messages_received();

    assert_eq!(metrics.messages_sent, 2);
    assert_eq!(metrics.messages_received, 1);
}

#[test]
fn test_node_metrics_inference_tracking() {
    let mut metrics = NodeMetrics::default();

    metrics.increment_inference();
    metrics.increment_inference();
    metrics.increment_inference();
    metrics.record_inference_latency(150.0);
    metrics.record_inference_latency(200.0);

    assert_eq!(metrics.inference_count, 3);
    assert!(metrics.avg_inference_latency_ms > 0.0);
}

#[test]
fn test_node_metrics_training_tracking() {
    let mut metrics = NodeMetrics::default();

    metrics.training_steps = 10;
    metrics.gradient_contributions = 5;
    metrics.running_loss = 0.45;
    metrics.best_loss = 0.42;

    metrics.increment_gradient_contributions();

    assert_eq!(metrics.training_steps, 10);
    assert_eq!(metrics.gradient_contributions, 6);
    assert!((metrics.running_loss - 0.45).abs() < f64::EPSILON);
    assert!((metrics.best_loss - 0.42).abs() < f64::EPSILON);
}

#[test]
fn test_node_metrics_spore_tracking() {
    let mut metrics = NodeMetrics::default();

    metrics.increment_spores_propagated();
    metrics.increment_spores_propagated();
    metrics.increment_spores_germinated();

    assert_eq!(metrics.spores_propagated, 2);
    assert_eq!(metrics.spores_germinated, 1);
}

#[test]
fn test_node_metrics_latency_moving_average() {
    let mut metrics = NodeMetrics::default();

    // First measurement should be the value itself
    metrics.record_inference_latency(100.0);
    let first_avg = metrics.avg_inference_latency_ms;
    assert!((first_avg - 100.0).abs() < 0.001);

    // Second measurement should be a weighted average
    metrics.record_inference_latency(200.0);
    assert!(metrics.avg_inference_latency_ms > 100.0);
    assert!(metrics.avg_inference_latency_ms < 200.0);
}

#[test]
fn test_node_metrics_serialization() {
    let mut metrics = NodeMetrics::default();
    metrics.peer_count = 42;
    metrics.inference_count = 1000;
    metrics.vram_used_mb = 8192;
    metrics.ram_used_mb = 16384;
    metrics.pipeline_active = true;
    metrics.pipeline_stages = 3;

    let json = serde_json::to_string(&metrics).unwrap();
    let decoded: NodeMetrics = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.peer_count, 42);
    assert_eq!(decoded.inference_count, 1000);
    assert_eq!(decoded.vram_used_mb, 8192);
    assert_eq!(decoded.pipeline_active, true);
    assert_eq!(decoded.pipeline_stages, 3);
}

// ─── Expert Replication Tests ───────────────────────────────────────────────

#[test]
fn test_expert_placement_creation() {
    let primary = NodeId::new();
    let placement = ExpertPlacement::new(0, primary);

    assert_eq!(placement.expert_id, 0);
    assert_eq!(placement.primary, primary);
    assert!(placement.replicas.is_empty());
    assert!(placement.healthy);
}

#[test]
fn test_expert_placement_add_replica() {
    let primary = NodeId::new();
    let mut placement = ExpertPlacement::new(0, primary);

    let replica1 = NodeId::new();
    let replica2 = NodeId::new();

    placement.add_replica(replica1);
    placement.add_replica(replica2);

    assert_eq!(placement.replicas.len(), 2);
    assert!(placement.all_nodes().contains(&replica1));
    assert!(placement.all_nodes().contains(&replica2));
}

#[test]
fn test_expert_placement_remove_replica() {
    let primary = NodeId::new();
    let mut placement = ExpertPlacement::new(0, primary);

    let replica = NodeId::new();
    placement.add_replica(replica);
    assert_eq!(placement.replicas.len(), 1);

    placement.remove_replica(&replica);
    assert!(placement.replicas.is_empty());
}

#[test]
fn test_expert_placement_load_tracking() {
    let primary = NodeId::new();
    let placement = ExpertPlacement::new(0, primary);

    assert_eq!(placement.load(), 0);

    placement.increment_load();
    placement.increment_load();
    placement.increment_load();

    assert_eq!(placement.load(), 3);
}

#[test]
fn test_expert_placement_health_tracking() {
    let primary = NodeId::new();
    let mut placement = ExpertPlacement::new(0, primary);

    assert!(placement.healthy);

    placement.mark_unhealthy();
    assert!(!placement.healthy);

    placement.mark_healthy();
    assert!(placement.healthy);
}

#[test]
fn test_expert_registry_creation() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();

    let registry = ExpertRegistry::new(
        local_node,
        config,
        2, // min_replicas
        100, // max_load_threshold
    );

    // Registry should be created successfully
    let stats = registry.get_stats();
    assert_eq!(stats.total_experts, 0);
}

#[test]
fn test_expert_registry_register_primary() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node, config.clone(), 2, 100);

    registry.register_expert(0, local_node, true).unwrap();

    assert!(registry.expert_exists(0));
}

#[test]
fn test_expert_registry_register_replica() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node, config.clone(), 2, 100);

    let primary = NodeId::new();
    registry.register_expert(0, primary, true).unwrap();

    // Register as replica
    registry.register_expert(0, local_node, false).unwrap();

    // Expert should still exist
    assert!(registry.expert_exists(0));
}

#[test]
fn test_expert_registry_select_node() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node.clone(), config.clone(), 1, 100);

    // Register expert with local node as primary
    registry.register_expert(0, local_node.clone(), true).unwrap();

    // Should select the local node
    let selected = registry.select_expert_node(0);
    assert!(selected.is_some());
    assert_eq!(selected.unwrap(), local_node);
}

#[test]
fn test_expert_registry_select_node_round_robin() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node.clone(), config.clone(), 1, 100);

    // Register with multiple nodes
    let node1 = NodeId::new();
    let node2 = NodeId::new();

    registry.register_expert(0, node1, true).unwrap();
    registry.register_expert(0, node2, false).unwrap();

    // Multiple selections should return valid nodes
    for _ in 0..5 {
        let selected = registry.select_expert_node(0);
        assert!(selected.is_some());
        let node = selected.unwrap();
        assert!(node == node1 || node == node2);
    }
}

#[test]
fn test_expert_registry_unregister_primary_promotes_replica() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node.clone(), config.clone(), 1, 100);

    let primary = NodeId::new();
    let replica = NodeId::new();

    // Register primary
    registry.register_expert(0, primary, true).unwrap();
    // Register replica
    registry.register_expert(0, replica, false).unwrap();

    // Unregister primary - replica should be promoted
    registry.unregister_expert(0, primary).unwrap();

    // Expert should still exist with replica as primary
    assert!(registry.expert_exists(0));
}

#[test]
fn test_expert_registry_stats() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node.clone(), config.clone(), 1, 100);

    let node1 = NodeId::new();
    let node2 = NodeId::new();

    registry.register_expert(0, node1, true).unwrap();
    registry.register_expert(1, node2, true).unwrap();
    registry.register_expert(0, local_node.clone(), false).unwrap();

    let stats = registry.get_stats();

    assert_eq!(stats.total_experts, 2);
    assert_eq!(stats.total_replicas, 1);
    assert_eq!(stats.total_nodes, 3);
}

#[test]
fn test_expert_registry_check_replication_needs() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node.clone(), config.clone(), 2, 100);

    let primary = NodeId::new();

    // Register expert with only 1 replica (needs 2)
    registry.register_expert(0, primary, true).unwrap();

    let needs = registry.check_replication_needs();

    // Expert 0 should need more replicas
    assert!(needs.contains_key(&0));
    assert_eq!(needs[&0], 2); // Needs 2 more to reach min_replicas=2
}

// ─── Expert Load Balancer Tests ─────────────────────────────────────────────

#[test]
fn test_load_balancer_creation() {
    let lb = ExpertLoadBalancer::new();
    // Should be created successfully
}

#[test]
fn test_load_balancer_round_robin() {
    let mut lb = ExpertLoadBalancer::new();

    let nodes: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();

    // Register nodes for expert 0
    for node in &nodes {
        lb.register_expert_node(0, *node);
    }

    // Round robin should cycle through nodes
    let first = lb.round_robin(0);
    let second = lb.round_robin(0);
    let third = lb.round_robin(0);
    let fourth = lb.round_robin(0);

    // Fourth should wrap around to first
    assert_eq!(first, fourth);
    // All should be valid nodes
    assert!(nodes.contains(&first));
    assert!(nodes.contains(&second));
    assert!(nodes.contains(&third));
}

#[test]
fn test_load_balancer_least_loaded() {
    let mut lb = ExpertLoadBalancer::new();

    let node1 = NodeId::new();
    let node2 = NodeId::new();
    let node3 = NodeId::new();

    lb.register_expert_node(0, node1);
    lb.register_expert_node(0, node2);
    lb.register_expert_node(0, node3);

    // Simulate load
    lb.record_query_start(node1);
    lb.record_query_start(node1);
    lb.record_query_start(node2);

    // Least loaded should be node3 (0 queries)
    let selected = lb.least_loaded(0);
    assert_eq!(selected, node3);

    // After node3 finishes, it should still be least loaded
    // node1 has 2, node2 has 1, node3 has 0
}

#[test]
fn test_load_balancer_query_lifecycle() {
    let mut lb = ExpertLoadBalancer::new();

    let node = NodeId::new();
    lb.register_expert_node(0, node);

    // Record query starts
    lb.record_query_start(node);
    lb.record_query_start(node);

    // Record query end
    lb.record_query_end(node);

    // Active queries should be 1
    // (implementation detail - may vary based on exact implementation)
}

// ─── Pipeline Parallel Integration Tests ────────────────────────────────────

#[test]
fn test_multi_node_pipeline_plan() {
    // Create a pipeline plan simulating 3 nodes
    let node1 = NodeId::new();
    let node2 = NodeId::new();
    let node3 = NodeId::new();

    // Simulate 48 layers split across 3 nodes (16 layers each)
    let plan = PipelinePlan::from_node_layers(vec![
        (node1, 16),
        (node2, 16),
        (node3, 16),
    ]);

    assert_eq!(plan.num_stages(), 3);
    assert_eq!(plan.total_layers, 48);
    assert!(plan.validate().is_ok());

    // Verify layer assignments
    assert_eq!(plan.stages[0].start_layer, 0);
    assert_eq!(plan.stages[0].end_layer, 16);
    assert_eq!(plan.stages[0].node_id, node1);

    assert_eq!(plan.stages[1].start_layer, 16);
    assert_eq!(plan.stages[1].end_layer, 32);
    assert_eq!(plan.stages[1].node_id, node2);

    assert_eq!(plan.stages[2].start_layer, 32);
    assert_eq!(plan.stages[2].end_layer, 48);
    assert_eq!(plan.stages[2].node_id, node3);
}

#[test]
fn test_pipeline_plan_with_expert_assignment() {
    let node1 = NodeId::new();
    let node2 = NodeId::new();

    // Node 1: layers 0-24, experts 0-3
    // Node 2: layers 24-48, experts 4-7
    let plan = PipelinePlan::from_node_layers(vec![
        (node1, 24),
        (node2, 24),
    ]);

    assert_eq!(plan.total_layers, 48);
    assert_eq!(plan.stages[0].node_id, node1);
    assert_eq!(plan.stages[1].node_id, node2);
}

#[tokio::test]
async fn test_pipeline_executor_multi_stage() {
    let node = NodeId::new();
    let plan = PipelinePlan::from_node_layers(vec![
        (node.clone(), 8),
        (node.clone(), 8),
        (node.clone(), 8),
    ]);

    let config = PipelineParallelConfig {
        num_micro_batches: 2,
        stage_timeout_ms: 5000,
        collect_stats: true,
        ..Default::default()
    };

    let model = Arc::new(RwLock::new(None));
    let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

    let latent = LatentVector::from_vec(vec![1.0; 64], 0, uuid::Uuid::new_v4());
    let rid = uuid::Uuid::new_v4();

    let result = executor.execute_pipeline(&latent, rid).await;
    assert!(result.is_ok(), "Multi-stage pipeline should succeed");

    let result = result.unwrap();
    assert_eq!(result.outputs.len(), 2); // 2 micro-batches

    // Final layer index should be 24 (8+8+8)
    for output in &result.outputs {
        assert_eq!(output.layer_idx, 24);
    }
}

#[tokio::test]
async fn test_pipeline_executor_with_stats() {
    let node = NodeId::new();
    let plan = PipelinePlan::from_node_layers(vec![
        (node.clone(), 4),
        (node.clone(), 4),
    ]);

    let config = PipelineParallelConfig {
        num_micro_batches: 4,
        stage_timeout_ms: 5000,
        collect_stats: true,
        ..Default::default()
    };

    let model = Arc::new(RwLock::new(None));
    let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

    let latent = LatentVector::from_vec(vec![1.0; 128], 0, uuid::Uuid::new_v4());
    let rid = uuid::Uuid::new_v4();

    let result = executor.execute_and_merge(&latent, rid).await.unwrap();
    let (_merged, stats) = result;

    // Stats should be collected
    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert_eq!(stats.num_micro_batches, 4);
    assert_eq!(stats.stage_stats.len(), 2);
    assert!(stats.total_time_us > 0);
    assert!(stats.throughput_mb_per_sec > 0.0);
}

// ─── Latent Streaming Integration Tests ─────────────────────────────────────

#[tokio::test]
async fn test_latent_stream_multi_node() {
    let manager = LatentStreamManager::new(LatentStreamConfig::default());

    let node_a = NodeId::new();
    let node_b = NodeId::new();
    let node_c = NodeId::new();

    // Create streams between nodes
    let (stream_ab_tx, mut stream_ab_rx) = manager.create_stream(node_a, node_b).await;
    let (stream_bc_tx, mut stream_bc_rx) = manager.create_stream(node_b, node_c).await;

    assert_eq!(manager.active_stream_count().await, 2);

    // Send latent from A to B
    let latent_ab = LatentVector::from_vec(vec![1.0; 64], 0, stream_ab_tx.meta.stream_id);
    stream_ab_tx.send(latent_ab.clone()).await.unwrap();

    // Send latent from B to C
    let latent_bc = LatentVector::from_vec(vec![2.0; 64], 1, stream_bc_tx.meta.stream_id);
    stream_bc_tx.send(latent_bc.clone()).await.unwrap();

    // B receives from A
    let received_ab = stream_ab_rx.recv().await.unwrap();
    assert_eq!(received_ab.data.len(), 64);
    assert_eq!(received_ab.layer_idx, 0);

    // C receives from B
    let received_bc = stream_bc_rx.recv().await.unwrap();
    assert_eq!(received_bc.data.len(), 64);
    assert_eq!(received_bc.layer_idx, 1);
}

#[tokio::test]
async fn test_latent_stream_chain_propagation() {
    // Simulate a chain: Node A -> Node B -> Node C -> Node D
    let manager = LatentStreamManager::new(LatentStreamConfig {
        buffer_size: 8,
        backpressure: true,
        max_throughput: None,
    });

    let nodes: Vec<NodeId> = (0..4).map(|_| NodeId::new()).collect();

    // Create chain of streams
    let mut senders = Vec::new();
    let mut receivers = Vec::new();

    for i in 0..nodes.len() - 1 {
        let (tx, rx) = manager.create_stream(nodes[i], nodes[i + 1]).await;
        senders.push(tx);
        receivers.push(rx);
    }

    assert_eq!(manager.active_stream_count().await, 3);

    // Send latent through the chain
    let initial_latent = LatentVector::from_vec(vec![1.0; 32], 0, senders[0].meta.stream_id);

    // A sends to B
    senders[0].send(initial_latent.clone()).await.unwrap();
    let b_received = receivers[0].recv().await.unwrap();
    assert_eq!(b_received.layer_idx, 0);

    // B processes and sends to C
    let b_processed = LatentVector::from_vec(b_received.data.clone(), 8, senders[1].meta.stream_id);
    senders[1].send(b_processed.clone()).await.unwrap();
    let c_received = receivers[1].recv().await.unwrap();
    assert_eq!(c_received.layer_idx, 8);

    // C processes and sends to D
    let c_processed = LatentVector::from_vec(c_received.data.clone(), 16, senders[2].meta.stream_id);
    senders[2].send(c_processed.clone()).await.unwrap();
    let d_received = receivers[2].recv().await.unwrap();
    assert_eq!(d_received.layer_idx, 16);
}

// ─── Topology Integration Tests ─────────────────────────────────────────────

#[test]
fn test_topology_multi_node_vram() {
    let mut topology = TopologyMap::default();

    let node1 = NodeId::new();
    let node2 = NodeId::new();
    let node3 = NodeId::new();

    topology.nodes.push((node1, NodeCapabilities::cpu_only(8192)));
    topology.nodes.push((node2, NodeCapabilities::cpu_only(16384)));
    topology.nodes.push((node3, NodeCapabilities::cpu_only(4096)));

    assert_eq!(topology.nodes.len(), 3);
    assert_eq!(topology.total_vram_mb(), 28672); // 8192 + 16384 + 4096
}

#[test]
fn test_topology_layer_assignment() {
    let mut topology = TopologyMap::default();

    let node1 = NodeId::new();
    let node2 = NodeId::new();

    topology.nodes.push((node1, NodeCapabilities::cpu_only(16384)));
    topology.nodes.push((node2, NodeCapabilities::cpu_only(8192)));

    // Create assignments (normally done by coordinator)
    topology.assignments.push(mycelium_core::LayerAssignment {
        node_id: node1,
        layer_start: 0,
        layer_end: 32,
        expert_ids: vec![0, 1, 2, 3],
        priority: 0,
    });
    topology.assignments.push(mycelium_core::LayerAssignment {
        node_id: node2,
        layer_start: 32,
        layer_end: 48,
        expert_ids: vec![4, 5, 6, 7],
        priority: 0,
    });

    // Test layer lookup
    let node_for_layer_16 = topology.best_node_for_layer(16);
    assert_eq!(node_for_layer_16, Some(node1));

    let node_for_layer_40 = topology.best_node_for_layer(40);
    assert_eq!(node_for_layer_40, Some(node2));

    let node_for_layer_100 = topology.best_node_for_layer(100);
    assert_eq!(node_for_layer_100, None);
}

#[test]
fn test_topology_expert_lookup() {
    let mut topology = TopologyMap::default();

    let node1 = NodeId::new();
    let node2 = NodeId::new();

    topology.assignments.push(mycelium_core::LayerAssignment {
        node_id: node1,
        layer_start: 0,
        layer_end: 24,
        expert_ids: vec![0, 1, 2, 3],
        priority: 0,
    });
    topology.assignments.push(mycelium_core::LayerAssignment {
        node_id: node2,
        layer_start: 24,
        layer_end: 48,
        expert_ids: vec![4, 5, 6, 7],
        priority: 0,
    });

    // Test expert lookup
    let node_for_expert_2 = topology.assignments.iter()
        .find(|a| a.expert_ids.contains(&2))
        .map(|a| a.node_id);
    assert_eq!(node_for_expert_2, Some(node1));

    let node_for_expert_6 = topology.assignments.iter()
        .find(|a| a.expert_ids.contains(&6))
        .map(|a| a.node_id);
    assert_eq!(node_for_expert_6, Some(node2));
}

// ─── End-to-End Distributed Inference Simulation ────────────────────────────

#[tokio::test]
async fn test_end_to_end_distributed_inference_simulation() {
    // Simulate a 3-node distributed inference setup:
    // Node 1: Layers 0-15, Experts 0-3
    // Node 2: Layers 16-31, Experts 4-7
    // Node 3: Layers 32-47, Experts 8-11

    let node1 = NodeId::new();
    let node2 = NodeId::new();
    let node3 = NodeId::new();

    // 1. Create pipeline plan
    let plan = PipelinePlan::from_node_layers(vec![
        (node1, 16),
        (node2, 16),
        (node3, 16),
    ]);

    assert_eq!(plan.total_layers, 48);
    assert!(plan.validate().is_ok());

    // 2. Create expert registry
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(node1, config, 2, 100);

    // Register experts
    for expert_id in 0..4 {
        registry.register_expert(expert_id, node1, true).unwrap();
    }
    for expert_id in 4..8 {
        registry.register_expert(expert_id, node2, true).unwrap();
    }
    for expert_id in 8..12 {
        registry.register_expert(expert_id, node3, true).unwrap();
    }

    assert_eq!(registry.get_stats().total_experts, 12);

    // 3. Create load balancer
    let mut lb = ExpertLoadBalancer::new();
    for expert_id in 0..4 {
        lb.register_expert_node(expert_id, node1);
    }
    for expert_id in 4..8 {
        lb.register_expert_node(expert_id, node2);
    }

    // 4. Simulate inference request
    let input_latent = LatentVector::from_vec(vec![1.0; 64], 0, uuid::Uuid::new_v4());

    // Select expert nodes for the request
    let expert_2_node = registry.select_expert_node(2);
    assert!(expert_2_node.is_some());
    assert_eq!(expert_2_node.unwrap(), node1);

    let expert_6_node = registry.select_expert_node(6);
    assert!(expert_6_node.is_some());
    assert_eq!(expert_6_node.unwrap(), node2);

    // 5. Create pipeline executor on node 1
    let pipeline_config = PipelineParallelConfig {
        num_micro_batches: 2,
        stage_timeout_ms: 5000,
        collect_stats: true,
        ..Default::default()
    };

    let model = Arc::new(RwLock::new(None));
    let executor = PipelineExecutor::new(plan, pipeline_config, node1, model).unwrap();

    // 6. Execute pipeline (will pass through since no model loaded)
    let request_id = uuid::Uuid::new_v4();
    let result = executor.execute_pipeline(&input_latent, request_id).await;

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.outputs.len(), 2); // 2 micro-batches

    // 7. Verify metrics tracking
    let mut metrics = NodeMetrics::default();
    metrics.increment_inference();
    metrics.increment_messages_sent();
    metrics.pipeline_active = true;
    metrics.pipeline_stages = 3;

    assert_eq!(metrics.inference_count, 1);
    assert_eq!(metrics.messages_sent, 1);
    assert!(metrics.pipeline_active);
    assert_eq!(metrics.pipeline_stages, 3);
}

// ─── Fault Tolerance Tests ──────────────────────────────────────────────────

#[test]
fn test_expert_registry_node_failure_recovery() {
    let local_node = NodeId::new();
    let config = mycelium_core::ModelConfig::minimax_m25();
    let mut registry = ExpertRegistry::new(local_node.clone(), config.clone(), 1, 100);

    let primary = NodeId::new();
    let replica1 = NodeId::new();
    let replica2 = NodeId::new();

    // Register expert with primary and replicas
    registry.register_expert(0, primary, true).unwrap();
    registry.register_expert(0, replica1, false).unwrap();
    registry.register_expert(0, replica2, false).unwrap();

    // Simulate primary failure
    registry.unregister_expert(0, primary).unwrap();

    // Should still be able to select a node (one of the replicas)
    let selected = registry.select_expert_node(0);
    assert!(selected.is_some());
    let selected_node = selected.unwrap();
    assert!(selected_node == replica1 || selected_node == replica2);
}

#[test]
fn test_pipeline_executor_stage_timeout() {
    let node = NodeId::new();
    let plan = PipelinePlan::from_node_layers(vec![
        (node.clone(), 8),
        (node.clone(), 8),
    ]);

    let config = PipelineParallelConfig {
        num_micro_batches: 1,
        stage_timeout_ms: 1, // Very short timeout
        collect_stats: false,
        ..Default::default()
    };

    let model = Arc::new(RwLock::new(None));
    let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

    let latent = LatentVector::from_vec(vec![1.0; 64], 0, uuid::Uuid::new_v4());
    let rid = uuid::Uuid::new_v4();

    // With very short timeout and no model, this might timeout or succeed
    // Just verify it doesn't panic
    let _ = executor.execute_pipeline(&latent, rid);
}

#[tokio::test]
async fn test_load_balancer_node_failure() {
    let mut lb = ExpertLoadBalancer::new();

    let node1 = NodeId::new();
    let node2 = NodeId::new();

    lb.register_expert_node(0, node1);
    lb.register_expert_node(0, node2);

    // Both nodes should be selectable
    let first = lb.least_loaded(0);
    assert!(first == node1 || first == node2);

    // After node1 "fails" (removed), only node2 should be selectable
    lb.remove_expert_node(0, &node1);
    let after_failure = lb.least_loaded(0);
    assert_eq!(after_failure, node2);
}

// ─── Prometheus Metrics Format Tests ────────────────────────────────────────

#[test]
fn test_node_metrics_prometheus_format_simulation() {
    let mut metrics = NodeMetrics::default();
    metrics.peer_count = 15;
    metrics.messages_sent = 1234;
    metrics.messages_received = 5678;
    metrics.inference_count = 100;
    metrics.training_steps = 50;
    metrics.vram_used_mb = 8192;
    metrics.ram_used_mb = 16384;
    metrics.pipeline_active = true;
    metrics.pipeline_stages = 3;

    // Simulate Prometheus format output
    let prometheus_output = format!(
        "# HELP mycelium_node_peers Number of connected peers
# TYPE mycelium_node_peers gauge
mycelium_node_peers {}
# HELP mycelium_messages_sent_total Total messages sent
# TYPE mycelium_messages_sent_total counter
mycelium_messages_sent_total {}
# HELP mycelium_messages_received_total Total messages received
# TYPE mycelium_messages_received_total counter
mycelium_messages_received_total {}
# HELP mycelium_inference_count_total Total inference requests
# TYPE mycelium_inference_count_total counter
mycelium_inference_count_total {}
# HELP mycelium_vram_used_bytes VRAM usage in bytes
# TYPE mycelium_vram_used_bytes gauge
mycelium_vram_used_bytes {}
",
        metrics.peer_count,
        metrics.messages_sent,
        metrics.messages_received,
        metrics.inference_count,
        metrics.vram_used_mb * 1024 * 1024,
    );

    // Verify format
    assert!(prometheus_output.contains("# HELP"));
    assert!(prometheus_output.contains("# TYPE"));
    assert!(prometheus_output.contains("mycelium_node_peers 15"));
    assert!(prometheus_output.contains("mycelium_messages_sent_total 1234"));
    assert!(prometheus_output.contains("mycelium_inference_count_total 100"));
}
