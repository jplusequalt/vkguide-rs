use anyhow::Result;
use ash::{Device, vk};

pub struct PoolSizeRatio {
    descriptor_type: vk::DescriptorType,
    ratio: f32,
}

impl PoolSizeRatio {
    pub fn new(descriptor_type: vk::DescriptorType, ratio: f32) -> Self {
        Self {
            descriptor_type,
            ratio,
        }
    }
}

#[derive(Default)]
pub struct DescriptorAllocator {
    pool: vk::DescriptorPool,
}

impl DescriptorAllocator {
    pub fn new(device: &Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Result<Self> {
        let pool_sizes = pool_ratios
            .iter()
            .map(|size| {
                vk::DescriptorPoolSize::default()
                    .descriptor_count(size.ratio as u32 * max_sets)
                    .ty(size.descriptor_type)
            })
            .collect::<Vec<_>>();

        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes[..]);

        let pool = unsafe { device.create_descriptor_pool(&info, None)? };
        Ok(Self { pool })
    }

    pub fn allocate_descriptor_sets(
        &self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = &[layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        let descriptor_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? };

        Ok(descriptor_set)
    }

    pub fn clear(&self, device: &Device) -> Result<()> {
        unsafe { device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())? };

        Ok(())
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

pub fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
    flags: vk::DescriptorSetLayoutCreateFlags,
) -> Result<vk::DescriptorSetLayout> {
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(flags)
        .bindings(bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };

    Ok(descriptor_set_layout)
}
