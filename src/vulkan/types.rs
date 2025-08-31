use anyhow::Result;
use ash::{Device, vk};
use bytemuck::{Zeroable, Pod};
use gpu_allocator::vulkan::Allocator;

use crate::vulkan::buffers::AllocatedBuffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_x: f32,
    pub normal: [f32; 4],
    pub uv_y: f32,
    pub color: [f32; 4],
}

#[derive(Default)]
pub struct GPUMeshBuffers {
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GPUPushConstants {
    pub world_matrix: [[f32; 4]; 4],
    pub vertex_buffer_address: vk::DeviceAddress,
}
