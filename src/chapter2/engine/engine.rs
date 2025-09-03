#![allow(dead_code)]

use anyhow::{Result, anyhow};
use ash::khr::{surface, synchronization2};
use ash::vk::{ImageAspectFlags, ImageUsageFlags};
use ash::{Device, Entry, Instance, ext::debug_utils, vk};
use bytemuck::{Pod, Zeroable};
use egui_ash_renderer::Renderer;
use glam::Vec4;
use gpu_allocator::vulkan::Allocator;
use log::*;
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use vkguide_rs::utils::fps::FPSCounter;
use vkguide_rs::vulkan::descriptors::{
    DescriptorAllocator, PoolSizeRatio, create_descriptor_set_layout,
};
use vkguide_rs::vulkan::pipelines::create_shader_module;
use winit::event::WindowEvent;
use winit::raw_window_handle::HasWindowHandle;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

use vkguide_rs::vulkan::images::{AllocatedImage, blit_image, create_image_view, transition_image};
use vkguide_rs::vulkan::queue::QueueFamilyIndices;
use vkguide_rs::vulkan::swapchain::SwapchainSupport;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

const PORTABILITY_MACOS_VERSION: u32 = vk::make_api_version(0, 1, 3, 216);

const DEVICE_EXTENSIONS: &[&CStr] = &[ash::khr::swapchain::NAME, synchronization2::NAME];

pub struct Engine {
    entry: Entry,
    instance: Instance,
    device: Device,
    state: EngineState,
    allocator: Option<Arc<Mutex<Allocator>>>,
    egui_renderer: Option<Renderer>,
    delta_time: Instant,
}

#[derive(Default)]
pub struct EngineState {
    debug_utils: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface_functions: Option<surface::Instance>,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    queue_indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_functions: Option<ash::khr::swapchain::Device>,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    frames: [FrameData; 2],
    swapchain_semaphore: Vec<vk::Semaphore>,
    render_semaphore: Vec<vk::Semaphore>,
    current_semaphore: usize,
    frame_number: usize,
    draw_image: AllocatedImage,
    descriptor_allocator: DescriptorAllocator,
    draw_image_descriptors: Vec<vk::DescriptorSet>,
    draw_image_descriptor_layout: vk::DescriptorSetLayout,
    gradient_pipeline_layout: vk::PipelineLayout,
    gradient_pipeline: vk::Pipeline,
    egui_state: Option<egui_winit::State>,
    fps: FPSCounter,
    c1: Vec4,
    c2: Vec4,
}

const FRAME_OVERLAP: usize = 2;

#[derive(Default)]
pub struct FrameData {
    command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    render_fence: vk::Fence,
}

impl Engine {
    pub fn create(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        let mut state = EngineState::default();
        state.fps = FPSCounter::new(60);

        let instance = create_instance(&entry, window, &mut state)?;

        state.surface_functions = Some(surface::Instance::new(&entry, &instance));
        state.surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };
        state.c1 = Vec4::new(1.0, 0.0, 0.0, 1.0);
        state.c2 = Vec4::new(1.0, 0.0, 1.0, 1.0);

        select_physical_device_and_queue_indices(&instance, &mut state)?;

        let device = create_device(&entry, &instance, &mut state)?;

        let allocator = Arc::new(Mutex::new(create_allocator(
            &instance,
            state.physical_device,
            &device,
        )?));

        create_swapchain(window, &instance, &device, &mut state)?;

        create_draw_image(
            &device,
            allocator.lock().as_deref_mut().unwrap(),
            &mut state,
        )?;

        create_commands(&device, &mut state)?;

        create_sync_objects(&device, &mut state)?;

        setup_descriptors(&device, &mut state)?;

        create_background_pipelines(&device, &mut state)?;

        let egui_renderer = setup_egui(window, &device, &allocator, &mut state)?;

        Ok(Self {
            entry,
            instance,
            device,
            state,
            allocator: Some(allocator),
            egui_renderer: Some(egui_renderer),
            delta_time: Instant::now(),
        })
    }

    pub fn egui_window_event(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self
            .state
            .egui_state
            .as_mut()
            .unwrap()
            .on_window_event(window, event);
    }

    // #region render

    pub fn render(&mut self, window: &Window) -> Result<()> {
        window.request_redraw();

        let new_time = Instant::now();
        let frame_time = new_time.duration_since(self.delta_time).as_secs_f32() * 1000.0;
        self.delta_time = new_time;

        self.state.fps.add_frame_time(frame_time);
        let fps_avg = self.state.fps.get_fps();

        let frame = self
            .state
            .frames
            .get(self.state.frame_number % FRAME_OVERLAP)
            .unwrap();

        // wait till the last frame is finished rendering before continuing
        unsafe {
            self.device
                .wait_for_fences(&[frame.render_fence], true, u64::MAX)?;

            self.device.reset_fences(&[frame.render_fence])?;
        };

        let (image_index, _) = unsafe {
            self.state
                .swapchain_functions
                .as_ref()
                .unwrap()
                .acquire_next_image(
                    self.state.swapchain,
                    u64::MAX,
                    self.state.swapchain_semaphore[self.state.current_semaphore],
                    vk::Fence::null(),
                )?
        };

        let cmd = frame.main_command_buffer;

        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };

        let gui_input = self
            .state
            .egui_state
            .as_mut()
            .unwrap()
            .take_egui_input(window);

        let egui_ctx = self.state.egui_state.as_ref().unwrap().egui_ctx().clone();

        egui_ctx.begin_pass(gui_input);
        egui::Window::new("Debug GUI").show(&egui_ctx, |ui| {
            ui.heading(format!("{:.2} fps", fps_avg));

            egui::ComboBox::from_label("1")
                .selected_text(format!("Color 1 {:?}", self.state.c1.to_string()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.state.c1, Vec4::new(1.0, 0.0, 0.0, 1.0), "Red");
                    ui.selectable_value(&mut self.state.c1, Vec4::new(0.0, 0.0, 1.0, 1.0), "Blue");
                });

            egui::ComboBox::from_label("2")
                .selected_text(format!("Color 2 {:?}", self.state.c2.to_string()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.state.c2,
                        Vec4::new(1.0, 0.0, 1.0, 1.0),
                        "Magenta",
                    );
                    ui.selectable_value(&mut self.state.c2, Vec4::new(0.0, 1.0, 0.0, 1.0), "Green");
                });
        });

        let egui::FullOutput {
            platform_output,
            shapes,
            textures_delta,
            pixels_per_point,
            ..
        } = egui_ctx.end_pass();

        self.state
            .egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(window, platform_output);

        let primitives = egui_ctx.tessellate(shapes, pixels_per_point);

        if !textures_delta.set.is_empty() {
            self.egui_renderer.as_mut().unwrap().set_textures(
                self.state.graphics_queue,
                frame.command_pool,
                textures_delta.set.as_slice(),
            )?;
        }

        let cmd_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(cmd, &cmd_begin_info)? };

        let image = self.state.swapchain_images[image_index as usize];

        // transition the draw image into a format we can use
        transition_image(
            &self.device,
            cmd,
            self.state.draw_image.image,
            ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        )?;

        draw_background(&self.device, cmd, self.state.c1, self.state.c2, &self.state)?;

        // transition swapchain image to transfer layouts
        transition_image(
            &self.device,
            cmd,
            self.state.draw_image.image,
            ImageAspectFlags::COLOR,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        transition_image(
            &self.device,
            cmd,
            image,
            ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        let extent = vk::Extent2D::default()
            .width(self.state.draw_image.extent.width)
            .height(self.state.draw_image.extent.height);

        // blit to swapchain
        blit_image(
            &self.device,
            cmd,
            self.state.draw_image.image,
            image,
            extent,
            extent,
        )?;

        transition_image(
            &self.device,
            cmd,
            image,
            ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.state.swapchain_image_views[image_index as usize])
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.state.swapchain_extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        unsafe { self.device.cmd_begin_rendering(cmd, &rendering_info) };

        self.egui_renderer.as_mut().unwrap().cmd_draw(
            cmd,
            self.state.swapchain_extent,
            pixels_per_point,
            &primitives,
        )?;

        unsafe { self.device.cmd_end_rendering(cmd) };

        // transition swapchain to present mode
        transition_image(
            &self.device,
            cmd,
            image,
            ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        )?;

        unsafe { self.device.end_command_buffer(cmd)? };

        let cmd_submit_info = &[vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];

        // use the swapchain_semaphore to ensure this command doesn't start executing until the swapchain image has been acquired
        let semaphore_wait_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(self.state.swapchain_semaphore[self.state.current_semaphore])
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)];

        // use the render_semaphore to signal when the rendering is done and it is safe to present to window
        // we use the current swapchain image index for this
        let semaphore_signal_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(self.state.render_semaphore[image_index as usize])
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)];

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(cmd_submit_info)
            .wait_semaphore_infos(semaphore_wait_info)
            .signal_semaphore_infos(semaphore_signal_info);

        let submit_infos = &[submit_info];

        unsafe {
            self.device.queue_submit2(
                self.state.graphics_queue,
                submit_infos,
                frame.render_fence,
            )?
        };

        let swapchains = &[self.state.swapchain];
        let image_indices = &[image_index];
        let wait_semaphores = &[self.state.render_semaphore[image_index as usize]];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swapchains)
            .wait_semaphores(wait_semaphores)
            .image_indices(image_indices);

        unsafe {
            self.state
                .swapchain_functions
                .as_mut()
                .unwrap()
                .queue_present(self.state.graphics_queue, &present_info)?
        };

        self.state.current_semaphore =
            (self.state.current_semaphore + 1) % self.state.swapchain_image_views.len();
        self.state.frame_number += 1;

        if !textures_delta.free.is_empty() {
            self.egui_renderer
                .as_mut()
                .unwrap()
                .free_textures(&textures_delta.free)?;
        }

        Ok(())
    }

    // #endregion

    fn destroy_swapchain(&mut self) {
        unsafe {
            self.state.descriptor_allocator.destroy(&self.device);

            self.state
                .swapchain_image_views
                .iter()
                .for_each(|i| self.device.destroy_image_view(*i, None));

            self.state
                .swapchain_functions
                .as_mut()
                .unwrap()
                .destroy_swapchain(self.state.swapchain, None);

            self.device
                .destroy_pipeline_layout(self.state.gradient_pipeline_layout, None);
            self.device
                .destroy_pipeline(self.state.gradient_pipeline, None);
        }
    }

    pub fn destroy(&mut self) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;

            self.state.draw_image.cleanup(
                &self.device,
                self.allocator
                    .as_mut()
                    .unwrap()
                    .lock()
                    .as_deref_mut()
                    .unwrap(),
            )?;

            self.state.egui_state = None;
            self.egui_renderer = None;

            self.state.frames.iter_mut().for_each(|f| {
                self.device.destroy_fence(f.render_fence, None);
            });

            for i in 0..self.state.swapchain_image_views.len() {
                self.device
                    .destroy_semaphore(self.state.render_semaphore[i], None);
                self.device
                    .destroy_semaphore(self.state.swapchain_semaphore[i], None);
            }

            self.destroy_swapchain();

            self.device
                .destroy_descriptor_set_layout(self.state.draw_image_descriptor_layout, None);

            self.state
                .frames
                .iter()
                .for_each(|f| self.device.destroy_command_pool(f.command_pool, None));

            self.allocator = None;

            self.device.destroy_device(None);
            self.state
                .surface_functions
                .as_mut()
                .unwrap()
                .destroy_surface(self.state.surface, None);

            if VALIDATION_ENABLED {
                self.state
                    .debug_utils
                    .as_mut()
                    .unwrap()
                    .destroy_debug_utils_messenger(
                        self.state.debug_messenger.take().unwrap(),
                        None,
                    );
            }

            self.instance.destroy_instance(None);
        }

        Ok(())
    }

    fn get_current_frame(&self) -> Option<&FrameData> {
        self.state
            .frames
            .get(self.state.frame_number % FRAME_OVERLAP)
    }
}

// #region Instance

fn create_instance(entry: &Entry, window: &Window, state: &mut EngineState) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::default()
        .application_name(c"chapter0")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"No engine")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        // use Vulkan 1.3
        .api_version(vk::API_VERSION_1_3);

    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    let layer_names = layers
        .iter()
        .map(|l| l.layer_name_as_c_str().unwrap())
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !layer_names.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported!"));
    }

    let layer_names_raw = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut extensions =
        ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();

    if VALIDATION_ENABLED {
        extensions.push(debug_utils::NAME.as_ptr());
    }

    let instance_version = unsafe { entry.try_enumerate_instance_version()? };
    if instance_version.is_none() {
        return Err(anyhow!("Could not retrieve Instance version"));
    }

    let flags =
        if cfg!(target_os = "macos") && instance_version.unwrap() >= PORTABILITY_MACOS_VERSION {
            info!("Enabling extensions for macOS portability.");
            // need these if using a non-conformant Vulkan implementation and vulkan version >= installed version
            // need this to enable the KHR_PORTABILITY_SUBSET_EXTENSION device
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

    let mut info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_layer_names(&layer_names_raw)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = unsafe { entry.create_instance(&info, None)? };

    // if validation enabled, add the debug info to enable custom logging
    if VALIDATION_ENABLED {
        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let debug_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&debug_info, None)? };

        state.debug_utils = Some(debug_utils);
        state.debug_messenger = Some(debug_messenger);
    }

    Ok(instance)
}

extern "system" fn debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    unsafe {
        use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

        let message = CStr::from_ptr((*p_callback_data).p_message);
        match flag {
            Flag::VERBOSE => log::debug!("{typ:?} - {message:?}"),
            Flag::INFO => log::info!("{typ:?} - {message:?}"),
            Flag::WARNING => log::warn!("{typ:?} - {message:?}"),
            _ => log::error!("{typ:?} - {message:?}"),
        }
        vk::FALSE
    }
}

// #endregion

// #region PhysicalDevice

fn select_physical_device_and_queue_indices(
    instance: &Instance,
    state: &mut EngineState,
) -> Result<()> {
    for physical_device in unsafe { instance.enumerate_physical_devices()? } {
        let mut properties = vk::PhysicalDeviceProperties2::default();
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut properties);
        }

        let mut is_suitable = properties.properties.api_version >= vk::API_VERSION_1_3;

        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let extension_names = extensions
            .iter()
            .map(|e| e.extension_name_as_c_str().unwrap())
            .collect::<HashSet<_>>();

        is_suitable &= DEVICE_EXTENSIONS
            .iter()
            .all(|e| extension_names.contains(*e));

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let indices = QueueFamilyIndices::find_queue_family_indices(
            physical_device,
            state.surface_functions.as_ref(),
            state.surface,
            queue_family_properties,
        )?;

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };

        is_suitable &= features13.dynamic_rendering == vk::TRUE
            && features13.synchronization2 == vk::TRUE
            && indices.graphics.is_some()
            && indices.present.is_some()
            && indices.transfer.is_some();

        if is_suitable {
            state.physical_device = physical_device;
            state.queue_indices = indices;

            info!(
                "Selected physical device {}",
                properties.properties.device_name_as_c_str()?.to_str()?
            );
            return Ok(());
        } else {
            warn!(
                "Skipping physical device {}",
                properties.properties.device_name_as_c_str()?.to_str()?
            );
        }
    }

    return Err(anyhow!("No suitable GPU found!"));
}

// #endregion

// #region Device

#[allow(unused_must_use)]
pub fn create_device(
    entry: &Entry,
    instance: &Instance,
    state: &mut EngineState,
) -> Result<Device> {
    let mut unique_indices = HashSet::new();
    unique_indices.insert(state.queue_indices.graphics);
    unique_indices.insert(state.queue_indices.present);
    unique_indices.insert(state.queue_indices.transfer);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*i.as_ref().unwrap())
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    let instance_version = unsafe { entry.try_enumerate_instance_version()? };
    if cfg!(target_os = "macos") && instance_version.unwrap() >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(ash::khr::portability_subset::NAME.as_ptr());
    }

    let mut vulkan_features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true)
        .dynamic_rendering(true);
    let mut vulkan_features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
        .descriptor_indexing(true)
        .buffer_device_address(true);

    let mut features = vk::PhysicalDeviceFeatures2::default();
    features = features.push_next(&mut vulkan_features_1_2);
    features = features.push_next(&mut vulkan_features_1_3);

    let info = vk::DeviceCreateInfo::default()
        .enabled_extension_names(&extensions)
        .queue_create_infos(&queue_infos)
        .push_next(&mut features);

    let device = unsafe { instance.create_device(state.physical_device, &info, None)? };

    unsafe {
        state.graphics_queue = device.get_device_queue(state.queue_indices.graphics.unwrap(), 0);
        state.transfer_queue = device.get_device_queue(state.queue_indices.transfer.unwrap(), 0);
        state.present_queue = device.get_device_queue(state.queue_indices.present.unwrap(), 0);
    };

    Ok(device)
}

// #endregion

// #region Swapchain

pub fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    state: &mut EngineState,
) -> Result<()> {
    let support = SwapchainSupport::get(
        state.physical_device,
        state.surface_functions.as_ref(),
        state.surface,
    )?;
    let surface_format = support.select_swapchain_surface_format();
    let present_modes = support.select_swapchain_present_mode();
    let extent = support.select_swapchain_extent(window);

    state.swapchain_extent = extent;
    state.swapchain_format = surface_format.format;

    // decide the number of images to use in the swapchain
    // best practice to set to 1 more than min
    let mut image_count = support.capabilities.min_image_count + 1;
    // if the max image count exists, check to see if the image count is > max count
    // if so, image count should be equal to the max
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if state.queue_indices.graphics != state.queue_indices.present {
        queue_family_indices.push(state.queue_indices.graphics.unwrap());
        queue_family_indices.push(state.queue_indices.present.unwrap());
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(state.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_modes)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_functions = ash::khr::swapchain::Device::new(instance, device);
    state.swapchain = unsafe { swapchain_functions.create_swapchain(&info, None)? };
    state.swapchain_images = unsafe { swapchain_functions.get_swapchain_images(state.swapchain)? };
    state.swapchain_functions = Some(swapchain_functions);

    state.swapchain_image_views = state
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                state.swapchain_format,
                vk::ImageAspectFlags::COLOR,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

pub fn create_draw_image(
    device: &Device,
    allocator: &mut Allocator,
    state: &mut EngineState,
) -> Result<()> {
    let extent = vk::Extent3D::default()
        .width(state.swapchain_extent.width)
        .height(state.swapchain_extent.height)
        .depth(1);

    let format = vk::Format::R16G16B16A16_SFLOAT;

    let usage = vk::ImageUsageFlags::TRANSFER_SRC
        | ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::COLOR_ATTACHMENT;

    state.draw_image = AllocatedImage::new(device, allocator, extent, usage, format)?;

    Ok(())
}

// #endregion

// #region Command

pub fn create_commands(device: &Device, state: &mut EngineState) -> Result<()> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(state.queue_indices.graphics.unwrap())
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    for i in 0..FRAME_OVERLAP {
        state.frames[i].command_pool =
            unsafe { device.create_command_pool(&command_pool_info, None)? };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(state.frames[i].command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        state.frames[i].main_command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)?[0] };
    }

    Ok(())
}

// #endregion

// #region Synchronization

pub fn create_sync_objects(device: &Device, state: &mut EngineState) -> Result<()> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    let semaphore_info = vk::SemaphoreCreateInfo::default();

    for i in 0..FRAME_OVERLAP {
        state.frames[i].render_fence = unsafe { device.create_fence(&fence_info, None)? };
    }

    for _ in 0..state.swapchain_image_views.len() {
        state
            .swapchain_semaphore
            .push(unsafe { device.create_semaphore(&semaphore_info, None)? });
        state
            .render_semaphore
            .push(unsafe { device.create_semaphore(&semaphore_info, None)? });
    }

    state.current_semaphore = 0;

    Ok(())
}

// #endregion

// #region Allocator

pub fn create_allocator(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
) -> Result<gpu_allocator::vulkan::Allocator> {
    let allocator_create_info = gpu_allocator::vulkan::AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings {
            log_memory_information: true,
            ..Default::default()
        },
        buffer_device_address: true,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    };

    let allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_info)?;
    Ok(allocator)
}

// #endregion

// #region Draw

pub fn draw_background(
    device: &Device,
    cmd: vk::CommandBuffer,
    c1: Vec4,
    c2: Vec4,
    state: &EngineState,
) -> Result<()> {
    unsafe {
        // bind the compute pipeline
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, state.gradient_pipeline);

        // bind the descriptor sets
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            state.gradient_pipeline_layout,
            0,
            &state.draw_image_descriptors[..],
            &[],
        );

        let push_constants = PushConstants {
            data1: c1.to_array(),
            data2: c2.to_array(),
            ..Default::default()
        };

        device.cmd_push_constants(
            cmd,
            state.gradient_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&[push_constants]),
        );

        device.cmd_dispatch(
            cmd,
            f32::ceil(state.draw_image.extent.width as f32 / 16.0) as u32,
            f32::ceil(state.draw_image.extent.height as f32 / 16.0) as u32,
            1,
        );
    }

    Ok(())
}

// #endregion

// #region Descriptors

pub fn setup_descriptors(device: &Device, state: &mut EngineState) -> Result<()> {
    let sizes = vec![PoolSizeRatio::new(vk::DescriptorType::STORAGE_IMAGE, 1f32)];

    let descriptor_allocator = DescriptorAllocator::new(device, 10, sizes)?;

    let image_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE);

    let layout = create_descriptor_set_layout(
        device,
        &[image_binding],
        vk::DescriptorSetLayoutCreateFlags::empty(),
    )?;

    // allocate a descriptor set for the draw image
    state.draw_image_descriptors = descriptor_allocator.allocate_descriptor_sets(device, layout)?;

    let image_info = &[vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(state.draw_image.image_view)];

    let image_write = vk::WriteDescriptorSet::default()
        .dst_set(state.draw_image_descriptors[0])
        .dst_binding(0)
        .image_info(image_info)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE);

    let image_writes = &[image_write];

    unsafe { device.update_descriptor_sets(image_writes, &[] as &[vk::CopyDescriptorSet]) };

    state.draw_image_descriptor_layout = layout;
    state.descriptor_allocator = descriptor_allocator;

    Ok(())
}

// #endregion

// #region Pipelines

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct PushConstants {
    data1: [f32; 4],
    data2: [f32; 4],
    data3: [f32; 4],
    data4: [f32; 4],
}

pub fn create_background_pipelines(device: &Device, state: &mut EngineState) -> Result<()> {
    let layouts = &[state.draw_image_descriptor_layout];

    let push_constant_ranges = &[vk::PushConstantRange::default()
        .offset(0)
        .size(std::mem::size_of::<PushConstants>() as u32)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

    let compute_layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(push_constant_ranges)
        .set_layouts(layouts);

    state.gradient_pipeline_layout =
        unsafe { device.create_pipeline_layout(&compute_layout_info, None)? };

    let compute_shader_src = include_bytes!("../../../shaders/out/gradient_comp.spv");
    let compute_module = create_shader_module(device, &compute_shader_src[..])?;

    let compute_shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .module(compute_module)
        .name(c"main")
        .stage(vk::ShaderStageFlags::COMPUTE);

    let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
        .layout(state.gradient_pipeline_layout)
        .stage(compute_shader_stage);

    let compute_pipeline_infos = &[compute_pipeline_info];

    state.gradient_pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), compute_pipeline_infos, None)
            .expect("failed to create compute pipeline")[0]
    };

    unsafe {
        device.destroy_shader_module(compute_module, None);
    }

    Ok(())
}

// #endregion

// #region egui

pub fn setup_egui(
    window: &Window,
    device: &Device,
    allocator: &Arc<Mutex<Allocator>>,
    state: &mut EngineState,
) -> Result<Renderer> {
    let gui_context = egui::Context::default();
    gui_context.set_pixels_per_point(window.scale_factor() as _);

    let viewport_id = gui_context.viewport_id();
    let gui_state = egui_winit::State::new(
        gui_context,
        viewport_id,
        &window,
        Some(window.scale_factor() as _),
        Some(winit::window::Theme::Dark),
        None,
    );

    egui_extras::install_image_loaders(gui_state.egui_ctx());

    let egui_renderer = Renderer::with_gpu_allocator(
        allocator.clone(),
        device.clone(),
        egui_ash_renderer::DynamicRendering {
            color_attachment_format: state.swapchain_format,
            depth_attachment_format: None,
        },
        egui_ash_renderer::Options {
            in_flight_frames: FRAME_OVERLAP,
            ..Default::default()
        },
    )?;

    state.egui_state = Some(gui_state);

    Ok(egui_renderer)
}

// #endregion
