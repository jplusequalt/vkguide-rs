use anyhow::{Result, anyhow};

use ash::{khr::surface, vk};

#[derive(Default)]
pub struct QueueFamilyIndices {
    pub graphics: Option<u32>,
    pub present: Option<u32>,
    pub transfer: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new(graphics: u32, present: u32, transfer: u32) -> Self {
        Self {
            graphics: Some(graphics),
            present: Some(present),
            transfer: Some(transfer),
        }
    }

    pub fn find_queue_family_indices(
        physical_device: vk::PhysicalDevice,
        surface_function: Option<&surface::Instance>,
        surface: vk::SurfaceKHR,
        queue_family_properties: Vec<vk::QueueFamilyProperties>,
    ) -> Result<Self> {
        let mut graphics = None;
        let mut present = None;
        let mut transfer = None;

        for (queue_family_index, queue_family) in queue_family_properties
            .iter()
            .filter(|family| family.queue_count > 0)
            .enumerate()
        {
            let has_graphics = queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let has_compute = queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let has_transfer = queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER);

            if graphics.is_none() && has_graphics && has_compute {
                graphics = Some(queue_family_index as u32);
            }

            if transfer.is_none() && has_transfer && !has_graphics && !has_compute {
                transfer = Some(queue_family_index as u32);
            }

            let present_support = unsafe {
                surface_function
                    .as_ref()
                    .unwrap()
                    .get_physical_device_surface_support(
                        physical_device,
                        queue_family_index as u32,
                        surface,
                    )
            }?;
            
            if present.is_none() && present_support {
                present = Some(queue_family_index as u32);
            }
        }

        Ok(Self {
            present,
            graphics,
            transfer,
        })
    }
}
