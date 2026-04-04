#!/usr/bin/env python3
"""Simple one-command builder for the C++ engine."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CPP_ENGINE_DIR = PROJECT_ROOT / "cpp_engine"


def _default_build_dir() -> Path:
    if sys.platform.startswith("win"):
        return CPP_ENGINE_DIR / "build-win"
    if sys.platform == "darwin":
        return CPP_ENGINE_DIR / "build-macos"
    return CPP_ENGINE_DIR / "build-linux"


def _find_cmake_command() -> list[str]:
    """Use system cmake if available, otherwise fall back to python -m cmake."""
    try:
        probe = subprocess.run(["cmake", "--version"], capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            return ["cmake"]
    except OSError:
        pass

    probe = subprocess.run([sys.executable, "-m", "cmake", "--version"], capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        return [sys.executable, "-m", "cmake"]

    raise RuntimeError(
        "CMake not found. Install CMake (system) or Python package cmake in this environment."
    )


def _run(cmd: list[str], cwd: Path | None = None, fail_hint: str | None = None) -> None:
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0:
        if fail_hint:
            print(fail_hint)
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def _run_pip(cmd: list[str]) -> None:
    pip_cmd = [sys.executable, "-m", "pip"] + cmd
    print("$ " + " ".join(pip_cmd))
    result = subprocess.run(pip_cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed with exit code {result.returncode}: {' '.join(pip_cmd)}")


def _install_requirements() -> None:
    if not REQUIREMENTS_FILE.exists():
        raise RuntimeError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    print("\n[0/1] Installing Python dependencies from requirements.txt")
    _run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"])
    _run_pip(["install", "-r", str(REQUIREMENTS_FILE)])


def _install_pytorch_cpu() -> None:
    print("\n[0/2] Installing CPU PyTorch and related packages")
    _run_pip(["install", "torch", "kagglehub", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    print("\n[0/3] Installing torch-geometric and ogb")
    _run_pip(["install", "torch-geometric", "ogb"])  # may require additional backend wheels on some platforms


def _print_artifacts(build_dir: Path) -> None:
    build_artifacts = list(build_dir.glob("cpp_engine*.pyd")) + list(build_dir.glob("cpp_engine*.so"))
    exp_artifacts = list((PROJECT_ROOT / "experiments").glob("cpp_engine*.pyd")) + list(
        (PROJECT_ROOT / "experiments").glob("cpp_engine*.so")
    )

    print("\nBuild artifacts:")
    if build_artifacts:
        for f in build_artifacts:
            print(f"  - {f}")
    else:
        print(f"  - No cpp_engine module found in {build_dir}")

    if exp_artifacts:
        print("\nInstalled artifacts (experiments):")
        for f in exp_artifacts:
            print(f"  - {f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the C++ engine with one command.")
    parser.add_argument("--clean", action="store_true", help="Delete selected build directory before configuring")
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Optional custom build directory (default is OS-specific: build-linux/build-win/build-macos)",
    )
    parser.add_argument(
        "--build-type",
        choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
        default="Release",
        help="CMake build type (default: Release)",
    )
    parser.add_argument("--jobs", type=int, default=0, help="Parallel build jobs (0 = CMake default)")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run cmake --install (installs module to experiments/ as configured in CMakeLists)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip cmake --install after building",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install Python dependencies from requirements.txt before building",
    )
    parser.add_argument(
        "--install-pytorch-cpu",
        action="store_true",
        help="Install CPU-only PyTorch, torchvision, torchaudio, torch-geometric, and ogb",
    )
    parser.add_argument(
        "--install-all",
        action="store_true",
        help="Install requirements.txt plus CPU PyTorch and torch-geometric/ogb",
    )
    args = parser.parse_args()
    if args.install_all:
        args.install_deps = True
        args.install_pytorch_cpu = True
        args.install = True

    if not args.install_deps and not args.install_all:
        args.install_deps = True

    if not args.no_install and not args.install:
        args.install = True
    if args.no_install:
        args.install = False

    build_dir = Path(args.build_dir).resolve() if args.build_dir else _default_build_dir()

    if not CPP_ENGINE_DIR.exists():
        print(f"ERROR: Missing directory: {CPP_ENGINE_DIR}")
        return 1

    windows_compiler_hint = ""
    if sys.platform.startswith("win"):
        has_msvc = shutil.which("cl") is not None
        has_gxx = shutil.which("g++") is not None
        if not (has_msvc or has_gxx):
            windows_compiler_hint = (
                "Hint: No C++ compiler detected in PATH. Install Visual Studio Build Tools "
                "(Desktop development with C++) or MinGW, then restart your terminal."
            )

    try:
        if args.install_deps or args.install_all:
            _install_requirements()

        if args.install_pytorch_cpu or args.install_all:
            _install_pytorch_cpu()

        cmake = _find_cmake_command()

        if args.clean and build_dir.exists():
            print(f"Removing existing build directory: {build_dir}")
            shutil.rmtree(build_dir)

        build_dir.mkdir(parents=True, exist_ok=True)

        print("\n[1/2] Configuring CMake")
        configure_cmd = cmake + [
            "-S",
            str(CPP_ENGINE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=" + args.build_type,
            "-DPython3_EXECUTABLE=" + sys.executable,
        ]
        _run(configure_cmd, fail_hint=windows_compiler_hint)

        print("\n[2/2] Building module")
        build_cmd = cmake + ["--build", str(build_dir), "--config", args.build_type]
        if args.jobs > 0:
            build_cmd += ["--parallel", str(args.jobs)]
        _run(build_cmd)

        if args.install:
            print("\n[3/3] Installing module")
            install_cmd = cmake + ["--install", str(build_dir), "--config", args.build_type]
            _run(install_cmd)

        print("\nBuild completed successfully.")
        _print_artifacts(build_dir)
        print("\nQuick import check:")
        print("  python test_import.py")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
