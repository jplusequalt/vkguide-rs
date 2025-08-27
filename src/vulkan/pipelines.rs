use anyhow::Result;
use ash::{Device, vk};

pub fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let mut cursor = std::io::Cursor::new(bytecode);
    let spirv_bytes = ash::util::read_spv(&mut cursor)?;
    let info = vk::ShaderModuleCreateInfo::default().code(&spirv_bytes);

    let shader_module = unsafe { device.create_shader_module(&info, None)? };

    Ok(shader_module)
}
