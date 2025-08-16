use anyhow::*;
use ash::{khr::surface, vk};
use winit::window::Window;

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    /// things like min/max number of images in swapchain, min/max image size, etc.
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    /// which surface formats (pixels, color space)
    pub formats: Vec<vk::SurfaceFormatKHR>,
    /// which presentation modes can be used
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub fn get(
        physical_device: vk::PhysicalDevice,
        surface_functions: Option<&surface::Instance>,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        if let Some(s_fn) = surface_functions {
            unsafe {
                Ok(Self {
                    capabilities: s_fn
                        .get_physical_device_surface_capabilities(physical_device, surface)?,
                    formats: s_fn.get_physical_device_surface_formats(physical_device, surface)?,
                    present_modes: s_fn
                        .get_physical_device_surface_present_modes(physical_device, surface)?,
                })
            }
        } else {
            Err(anyhow!("Surface does not exist!"))
        }
    }

    pub fn select_swapchain_surface_format(&self) -> vk::SurfaceFormatKHR {
        self.formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| self.formats[0])
    }

    pub fn select_swapchain_present_mode(&self) -> vk::PresentModeKHR {
        self.present_modes
            .iter()
            .cloned()
            .find(|m| *m == vk::PresentModeKHR::FIFO)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    pub fn select_swapchain_extent(&self, window: &Window) -> vk::Extent2D {
        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent
        } else {
            vk::Extent2D::default()
                .width(window.inner_size().width.clamp(
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                ))
                .height(window.inner_size().height.clamp(
                    self.capabilities.min_image_extent.height,
                    self.capabilities.max_image_extent.height,
                ))
        }
    }
}
