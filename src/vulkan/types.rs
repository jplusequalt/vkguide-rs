use anyhow::Result;
use ash::{Device, vk};
use glam::Vec3;
use gpu_allocator::vulkan::Allocator;

use crate::vulkan::buffers::AllocatedBuffer;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: [f32; 4],
    pub uv_x: f32,
    pub normal: [f32; 4],
    pub uv_y: f32,
    pub color: [f32; 4],
}

pub struct GPUMeshBuffers<'a> {
    pub index_buffer: AllocatedBuffer<'a>,
    pub vertex_buffer: AllocatedBuffer<'a>,
    pub vertex_buffer_address: vk::DeviceAddress,
}

impl<'a> GPUMeshBuffers<'a> {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        vertex_data: Vec<Vertex>,
        num_verts: usize,
        indices: Vec<u32>,
        num_indices: usize,
    ) -> Result<Self> {
        let vert_buffer_size = (num_verts * std::mem::size_of::<Vertex>()) as u64;
        let index_buffer_size = (num_indices * std::mem::size_of::<u32>()) as u64;

        let vertex_buffer = AllocatedBuffer::new(
            device,
            allocator,
            vert_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;

        let index_buffer = AllocatedBuffer::new(
            device,
            allocator,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;

        let buffer_address_info =
            vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer);
        let vertex_buffer_address =
            unsafe { device.get_buffer_device_address(&buffer_address_info) };

        let mut staging = AllocatedBuffer::new(
            device,
            allocator,
            vert_buffer_size + index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;

        presser::copy_from_slice_to_offset(&vertex_data[..], &mut staging.allocation, 0)?;
        presser::copy_from_slice_to_offset(&indices[..], &mut staging.allocation, index_buffer_size as usize)?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            vertex_buffer_address,
        })
    }
}

pub struct GPUPushConstants {
    pub world_matrix: [[f32; 4]; 4],
    pub vertex_buffer_address: vk::DeviceAddress,
}
