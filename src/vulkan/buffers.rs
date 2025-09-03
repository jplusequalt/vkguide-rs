use anyhow::Result;
use ash::{Device, vk};
use gpu_allocator::vulkan::{Allocation, Allocator};

#[derive(Default)]
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

impl AllocatedBuffer {
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
            linear: true,
            name: "buffer",
        };

        let allocation = allocator.allocate(&allocation_info)?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? }

        Ok(Self { buffer, allocation })
    }

    pub fn cleanup(&mut self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        let buffer = std::mem::take(self);
        allocator.free(buffer.allocation)?;

        unsafe {
            device.destroy_buffer(buffer.buffer, None);
        }

        Ok(())
    }
}
