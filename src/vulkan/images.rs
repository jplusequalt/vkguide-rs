use anyhow::*;

use ash::{Device, vk};
use gpu_allocator::vulkan::{Allocation, Allocator};

#[derive(Default)]
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
}

impl AllocatedImage {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
        format: vk::Format,
    ) -> Result<Self> {
        let create_info = vk::ImageCreateInfo::default()
            .extent(extent)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .mip_levels(1)
            .image_type(vk::ImageType::TYPE_2D)
            .samples(vk::SampleCountFlags::TYPE_1)
            .array_layers(1)
            .usage(usage);

        let image = unsafe { device.create_image(&create_info, None)? };

        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation_info = gpu_allocator::vulkan::AllocationCreateDesc {
            location: gpu_allocator::MemoryLocation::GpuOnly,
            requirements,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            linear: false,
            name: "image",
        };

        let allocation = allocator.allocate(&allocation_info)?;

        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let image_view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;

        Ok(Self {
            image,
            image_view,
            allocation,
            extent,
            format,
        })
    }

    pub fn cleanup(
        &mut self,
        device: &Device,
        allocator: &mut Allocator,
    ) -> Result<()> {
        let image = std::mem::take(self);
        allocator.free(image.allocation)?;

        unsafe {
            device.destroy_image_view(image.image_view, None);
            device.destroy_image(image.image, None);
        }

        Ok(())
    }
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

pub fn blit_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
) -> Result<()> {
    let src_offsets = [
        vk::Offset3D::default(),
        vk::Offset3D {
            x: src_size.width as i32,
            y: src_size.height as i32,
            z: 1,
        },
    ];

    let dst_offsets = [
        vk::Offset3D::default(),
        vk::Offset3D {
            x: dst_size.width as i32,
            y: dst_size.height as i32,
            z: 1,
        },
    ];

    let subresource_layer = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0);

    let blit_region = vk::ImageBlit2::default()
        .src_offsets(src_offsets)
        .dst_offsets(dst_offsets)
        .src_subresource(subresource_layer)
        .dst_subresource(subresource_layer);

    let regions = &[blit_region];
    let blit_info = vk::BlitImageInfo2::default()
        .src_image(src)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(dst)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(regions);

    unsafe {
        device.cmd_blit_image2(cmd, &blit_info);
    }

    Ok(())
}
