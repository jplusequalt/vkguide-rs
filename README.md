# vkguide-rs

This is a Rust implementation of [vkguide](https://vkguide.dev). Note that some bits are different from the original C++ guide as I also referenced other tutorials/resources in the process.

The core stack used in these tutorials is:

- [ash](https://github.com/ash-rs/ash) for Vulkan bindings
- [egui](https://github.com/emilk/egui)/[egui-ash-renderer](https://github.com/adrien-ben/egui-ash-renderer) for GUI support
- [gpu_allocator](https://github.com/Traverse-Research/gpu-allocator) for GPU memory management

You'll also need to have the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) installed on your machine to run this code.

## Running the chapters

The shaders need to be built first before running anything. The [compile_shaders.py](./compile_shaders.py) script will take care of this for you. This script assumes you installed the Vulkan SDK and thus have `glslc.exe` for compiling the GLSL into SPIR-V.

```bash
python compile_shaders.py
```

To run the code, specify a chapter with the `--bin` flag, e.g.:

```bash
cargo run --bin chapter1
```

## Disclaimer

There are no guarantees that this code works on any platforms other than Windows 10. Feel free to create a pull request to add cross-compatibility if you want, but don't expect any help from me on this front.
