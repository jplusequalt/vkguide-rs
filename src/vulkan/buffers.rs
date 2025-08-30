use anyhow::Result;
use ash::{Device, vk};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};

pub struct AllocatedBuffer<'a> {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub allocation_info: AllocationCreateDesc<'a>,
}

impl<'a> AllocatedBuffer<'a> {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: gpu_allocator::MemoryLocation,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default().usage(usage).size(size);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_info = gpu_allocator::vulkan::AllocationCreateDesc {
            location,
            requirements,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            linear: false,
            name: "test",
        };

        let allocation = allocator.allocate(&allocation_info)?;

        Ok(Self {
            buffer,
            allocation,
            allocation_info,
        })
    }
}