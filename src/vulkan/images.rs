use anyhow::*;

use ash::{Device, vk};
use gpu_allocator::vulkan::Allocation;

pub struct AllocatedImage {
    image: vk::Image,
    image_view: vk::ImageView,
    allocation: Allocation,
    extent: vk::Extent3D,
    format: vk::Format,
}

pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let components = vk::ComponentMapping::default()
        .r(vk::ComponentSwizzle::IDENTITY)
        .g(vk::ComponentSwizzle::IDENTITY)
        .b(vk::ComponentSwizzle::IDENTITY)
        .a(vk::ComponentSwizzle::IDENTITY);

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(components)
        .subresource_range(subresource_range);

    let image_view = unsafe { device.create_image_view(&info, None)? };

    Ok(image_view)
}

pub fn transition_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    image_aspect_flags: vk::ImageAspectFlags,
    current_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(image_aspect_flags)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS);

    let image_barrier = vk::ImageMemoryBarrier2::default()
        // stop all commands when they arrive at the stage
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .old_layout(current_layout)
        .new_layout(new_layout)
        .subresource_range(subresource_range)
        .image(image);

    let image_barriers = [image_barrier];
    let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);

    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dependency_info);
    }

    Ok(())
}
