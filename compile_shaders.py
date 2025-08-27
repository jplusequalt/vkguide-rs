import sys
import subprocess
import os
from pathlib import Path

folder = Path("./shaders/")

if not folder.is_dir():
    print(f"Error: path does not exist ./shaders/")
    sys.exit(1)

files_to_compile = [str(p.absolute()) for p in folder.glob("*") if p.is_file()]

if len(files_to_compile) == 0:
    print("No shaders to compile")
    sys.exit(1)

VULKAN_SDK = "VULKAN_SDK"

def compile_shader(path: Path):
    print(f"Compiling {path} ...")
    try:
        split = str(path).split(".")
        shader_type = split[1]

        path_segs = split[0].split("\\")
        parent_path = "/".join(path_segs[:-1])
        name = path_segs[-1]
        # you may not have this version of the sdk installed, so make sure to have the sdk in your path
        vulkan_path = os.environ.get(VULKAN_SDK) or "C:/VulkanSDK/1.4.321.1"
        subprocess.run(
            [
                f"{vulkan_path}/Bin/glslc.exe",
                str(path),
                "-o",
                f"{parent_path}/out/{name}_{shader_type}.spv",
            ],
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile {path}! Return code {e.returncode}")

[compile_shader(path) for path in files_to_compile]
