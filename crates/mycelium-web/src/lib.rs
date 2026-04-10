//! # Mycelium Web — WebAssembly Node
//!
//! Browser-based lightweight mycelium node using WebGPU for compute.
//! Provides:
//! - WebGPU-accelerated latent operations
//! - Matrix multiplication via WGSL shaders
//! - Lightweight inference in browser
//! - P2P connectivity (via WebRTC, future)
//!
//! Build with: `wasm-pack build --target web crates/mycelium-web`

use mycelium_core::{LatentVector, ModelConfig};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    Maintain, MapMode, ShaderModuleDescriptor, ShaderSource,
};
use wasm_bindgen::prelude::*;

// ─── WebGPU Compute Engine ─────────────────────────────────────────────────

/// WebGPU resources for compute operations.
struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("mycelium-webgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .ok()?;

        info!("WebGPU initialized: {:?}", adapter.get_info());
        Some(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

/// Shared state for the WASM module.
#[wasm_bindgen]
pub struct MyceliumWeb {
    gpu: Option<Arc<Mutex<GpuContext>>>,
    config: ModelConfig,
    shader_modules: std::collections::HashMap<String, wgpu::ShaderModule>,
}

#[wasm_bindgen]
impl MyceliumWeb {
    /// Create a new Mycelium web node.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Set up wasm logger
        let _ = console_error_panic_hook::set_once();

        info!("Mycelium Web node created");
        Self {
            gpu: None,
            config: ModelConfig::minimax_m25(),
            shader_modules: std::collections::HashMap::new(),
        }
    }

    /// Initialize WebGPU compute.
    pub async fn init_gpu(&mut self) -> Result<bool, JsError> {
        info!("Initializing WebGPU...");
        match GpuContext::new().await {
            Some(ctx) => {
                self.gpu = Some(Arc::new(Mutex::new(ctx)));
                info!("WebGPU initialized successfully");
                Ok(true)
            }
            None => {
                warn!("WebGPU not available, falling back to CPU");
                Ok(false)
            }
        }
    }

    /// Get node capabilities (for browser).
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> JsValue {
        let caps = serde_json::json!({
            "target": "browser",
            "compute": "WebGPU",
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "num_experts": self.config.num_experts,
            "context_length": self.config.max_context,
        });
        serde_wasm_bindgen::to_value(&caps).unwrap_or(JsValue::NULL)
    }

    /// Run latent vector interpolation (lerp) on GPU.
    pub async fn latent_lerp(
        &self,
        data_a: &[f32],
        data_b: &[f32],
        t: f32,
    ) -> Result<Vec<f32>, JsError> {
        let dim = data_a.len();
        if dim != data_b.len() {
            return Err(JsError::new("Latent vectors must have same dimension"));
        }

        // If GPU available, use compute shader
        if let Some(gpu) = &self.gpu {
            return self.gpu_lerp(gpu, data_a, data_b, t, dim as u32).await;
        }

        // CPU fallback
        let result: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(&a, &b)| a * (1.0 - t) + b * t)
            .collect();

        Ok(result)
    }

    /// Run matrix multiplication on GPU.
    pub async fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<Vec<f32>, JsError> {
        if let Some(gpu) = &self.gpu {
            return self.gpu_matmul(gpu, a, b, m, k, n).await;
        }

        // CPU fallback
        let mut c = vec![0.0f32; (m * n) as usize];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[(i * k + l) as usize] * b[(l * n + j) as usize];
                }
                c[(i * n + j) as usize] = sum;
            }
        }
        Ok(c)
    }

    /// Apply SiLU activation on GPU.
    pub async fn silu_activation(&self, input: &[f32]) -> Result<Vec<f32>, JsError> {
        if let Some(gpu) = &self.gpu {
            return self.gpu_activation(gpu, input, "silu").await;
        }

        // CPU fallback
        Ok(input
            .iter()
            .map(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect())
    }

    /// RMS normalization on GPU.
    pub async fn rms_norm(&self, input: &[f32], eps: f32) -> Result<Vec<f32>, JsError> {
        if let Some(gpu) = &self.gpu {
            return self.gpu_rms_norm(gpu, input, eps).await;
        }

        // CPU fallback
        let dim = input.len();
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        Ok(input.iter().map(|x| x / (rms + eps)).collect())
    }

    /// Create a latent vector from raw data.
    pub fn create_latent(&self, data: &[f32], layer: usize) -> Result<JsValue, JsError> {
        let latent = LatentVector::from_vec(data.to_vec(), layer, uuid::Uuid::new_v4());
        let json = serde_json::to_string(&latent).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(json.into())
    }

    /// Decode a latent vector from JSON.
    pub fn decode_latent(&self, json: &str) -> Result<JsValue, JsError> {
        let latent: LatentVector =
            serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))?;
        let data = serde_json::json!({
            "dim": latent.data.len(),
            "layer": latent.layer_idx,
            "first_10": &latent.data[..latent.data.len().min(10)],
        });
        Ok(serde_wasm_bindgen::to_value(&data).unwrap_or(JsValue::NULL))
    }

    /// Get model configuration.
    #[wasm_bindgen(getter)]
    pub fn model_config(&self) -> JsValue {
        let config = serde_json::json!({
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "num_experts": self.config.num_experts,
            "top_k_experts": self.config.top_k_experts,
            "context_length": self.config.max_context,
            "vram_estimate_mb": self.config.vram_estimate_mb(4),
        });
        serde_wasm_bindgen::to_value(&config).unwrap_or(JsValue::NULL)
    }

    /// Generate a random latent vector (for testing).
    pub fn random_latent(&self, dim: usize, _layer: usize) -> Result<Vec<f32>, JsError> {
        use js_sys::Math;
        Ok((0..dim).map(|_| (Math::random() as f32) * 2.0 - 1.0).collect())
    }

    /// Get the status of the WebGPU node.
    pub fn status(&self) -> JsValue {
        let status = serde_json::json!({
            "gpu_available": self.gpu.is_some(),
            "model": "minimax-m2.5",
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "num_experts": self.config.num_experts,
            "version": env!("CARGO_PKG_VERSION"),
        });
        serde_wasm_bindgen::to_value(&status).unwrap_or(JsValue::NULL)
    }
}

impl MyceliumWeb {
    // ─── GPU Compute Methods ───────────────────────────────────────────────

    async fn gpu_lerp(
        &self,
        gpu: &Arc<Mutex<GpuContext>>,
        data_a: &[f32],
        data_b: &[f32],
        t: f32,
        dim: u32,
    ) -> Result<Vec<f32>, JsError> {
        let guard = gpu.lock().unwrap();
        let device = &guard.device;
        let queue = &guard.queue;

        // Create buffers
        let buf_a = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("lerp_input_a"),
            contents: bytemuck::cast_slice(data_a),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let buf_b = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("lerp_input_b"),
            contents: bytemuck::cast_slice(data_b),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lerp_output"),
            size: (data_a.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Load shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("latent_ops"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/latent_ops.wgsl").into()),
        });

        // Create bind group layout and pipeline
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lerp_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lerp_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("lerp_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create uniform buffer for params
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Params {
            dim: u32,
            operation: u32,
            t: f32,
            scale: f32,
        }
        // Safety: Params is repr(C) and contains only POD types
        unsafe impl bytemuck::Pod for Params {}
        unsafe impl bytemuck::Zeroable for Params {}

        let params = Params {
            dim,
            operation: 0, // lerp
            t,
            scale: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("lerp_uniforms"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lerp_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode commands
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("lerp_encoder") });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lerp_compute_pass"),
                ..Default::default()
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((dim + 255) / 256, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        device.poll(Maintain::Wait);

        // Read back results
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(Maintain::Wait);
        rx.await.map_err(|e| JsError::new(&format!("Failed to read GPU buffer: {:?}", e)))??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Unmap buffer
        drop(data);
        output_buffer.unmap();

        Ok(result)
    }

    async fn gpu_matmul(
        &self,
        _gpu: &Arc<Mutex<GpuContext>>,
        _a: &[f32],
        _b: &[f32],
        _m: u32,
        _k: u32,
        _n: u32,
    ) -> Result<Vec<f32>, JsError> {
        // TODO: Implement full matmul with WebGPU
        // For now, return CPU result
        Err(JsError::new("GPU matmul not yet implemented"))
    }

    async fn gpu_activation(
        &self,
        gpu: &Arc<Mutex<GpuContext>>,
        input: &[f32],
        _activation: &str,
    ) -> Result<Vec<f32>, JsError> {
        // Use the same lerp infrastructure but with operation=4 (silu)
        let dim = input.len() as u32;
        let guard = gpu.lock().unwrap();
        let device = &guard.device;
        let queue = &guard.queue;

        let buf_input = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("activation_input"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("activation_output"),
            size: (input.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("latent_ops"),
            source: ShaderSource::Wgsl(include_str!("../../../shaders/latent_ops.wgsl").into()),
        });

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct Params {
            dim: u32,
            operation: u32,
            t: f32,
            scale: f32,
        }
        // Safety: Params is repr(C) and contains only POD types
        unsafe impl bytemuck::Pod for Params {}
        unsafe impl bytemuck::Zeroable for Params {}

        let params = Params {
            dim,
            operation: 4, // silu
            t: 0.0,
            scale: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("activation_uniforms"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("activation_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("activation_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_input.as_entire_binding(), // Use same buffer for both inputs
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("activation_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("activation_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("activation_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("activation_compute_pass"),
                ..Default::default()
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((dim + 255) / 256, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        device.poll(Maintain::Wait);

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(Maintain::Wait);
        rx.await.map_err(|e| JsError::new(&format!("Failed to read GPU buffer: {:?}", e)))??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        output_buffer.unmap();

        Ok(result)
    }

    async fn gpu_rms_norm(
        &self,
        _gpu: &Arc<Mutex<GpuContext>>,
        input: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, JsError> {
        // CPU fallback for now - RMSNorm requires reduction which is complex in WGSL
        let dim = input.len();
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        Ok(input.iter().map(|x| x / (rms + eps)).collect())
    }
}

// ─── WASM Initialization ────────────────────────────────────────────────────

/// Initialize the WASM module with console logging.
#[wasm_bindgen(start)]
pub fn start() {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize tracing for WASM
    tracing_wasm::set_as_global_default();

    info!("Mycelium Web module initialized");
}
