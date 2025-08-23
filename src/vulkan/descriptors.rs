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

pub fn create_descriptor_pool(
    device: &Device,
    max_sets: u32,
    pool_ratios: Vec<PoolSizeRatio>,
) -> Result<vk::DescriptorPool> {
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

    Ok(pool)
}

pub fn allocate_descriptor_sets(device: &Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = &[layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? };
    
    Ok(descriptor_set)
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
