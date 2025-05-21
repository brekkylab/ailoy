import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildCommand(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Collect possible install dirs from CMakePresets.json
        root_path = Path(__file__).parent.parent.parent
        cmake_preset_json_path = root_path / "CMakePresets.json"
        cmake_prefix_paths = []
        with open(cmake_preset_json_path) as f:
            cmake_presets = json.load(f)
            for preset in cmake_presets["configurePresets"]:
                install_dir = (
                    preset["installDir"]
                    .replace("${sourceDir}", str(root_path))
                    .replace("${presetName}", preset["name"])
                )
                cmake_prefix_paths.append(f"{install_dir}/lib/cmake/ailoy")
        cmake_prefix_path = ";".join(cmake_prefix_paths)

        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DSETUPTOOLS_EXTDIR={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        cfg = "Debug" if self.debug else "Release"
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(
            [
                "cmake",
                "--build",
                ".",
                "--target",
                "ailoy_py",
                "--parallel",
                f"{multiprocessing.cpu_count()}",
                "--config",
                cfg,
            ],
            cwd=build_temp,
        )


setup(
    name="ailoy",
    version="0.0.1",
    packages=["ailoy"],
    package_dir={"": "."},
    ext_modules=[CMakeExtension("ailoy.ailoy_py")],
    cmdclass={
        "build_ext": CMakeBuildCommand,
    },
    zip_safe=False,
)
