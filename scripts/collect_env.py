from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return {"found": False, "command": args}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"found": True, "command": args, "error": str(exc)}

    return {
        "found": True,
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def powershell_version_info(path: Path) -> dict[str, Any] | None:
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps or not path.exists():
        return None

    script = (
        "$item = Get-Item -LiteralPath '" + str(path).replace("'", "''") + "'; "
        "[PSCustomObject]@{"
        "Path=$item.FullName;"
        "ProductVersion=$item.VersionInfo.ProductVersion;"
        "FileVersion=$item.VersionInfo.FileVersion"
        "} | ConvertTo-Json -Compress"
    )
    result = run_command([ps, "-NoProfile", "-Command", script])
    if result.get("returncode") != 0 or not result.get("stdout"):
        return None
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"path": str(path), "raw": result["stdout"]}


def detect_os() -> dict[str, Any]:
    data: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }

    if os.name == "nt":
        try:
            import winreg  # type: ignore

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                data["product_name"] = winreg.QueryValueEx(key, "ProductName")[0]
                data["display_version"] = winreg.QueryValueEx(key, "DisplayVersion")[0]
                data["current_build"] = winreg.QueryValueEx(key, "CurrentBuild")[0]
                data["ubr"] = winreg.QueryValueEx(key, "UBR")[0]
        except Exception as exc:  # pragma: no cover - defensive path
            data["registry_error"] = str(exc)

    return data


def detect_python() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }


def detect_module(name: str, version_attr: str = "__version__") -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}

    return {
        "importable": True,
        "version": getattr(module, version_attr, None),
        "file": getattr(module, "__file__", None),
    }


def detect_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}

    data: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        data["devices"] = devices

    return data


def detect_ninja() -> dict[str, Any]:
    candidate = None
    preferred = Path(sys.executable).resolve().parent / ("ninja.exe" if os.name == "nt" else "ninja")
    if preferred.exists():
        candidate = str(preferred)
    else:
        candidate = shutil.which("ninja")

    if not candidate:
        return {"found": False}

    result = run_command([candidate, "--version"])
    return {
        "found": True,
        "path": candidate,
        "version": result.get("stdout") or result.get("stderr"),
    }


def nvcc_candidates() -> list[tuple[str, Path]]:
    from_env = os.environ.get("CUDA_HOME")
    candidates: list[tuple[str, Path]] = []
    if from_env:
        candidates.append(("CUDA_HOME", Path(from_env) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")))
    if os.name == "nt":
        toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if toolkit_root.exists():
            for version_dir in sorted([p for p in toolkit_root.iterdir() if p.is_dir()], reverse=True):
                candidates.append((f"installed:{version_dir.name}", version_dir / "bin" / "nvcc.exe"))
    which = shutil.which("nvcc")
    if which:
        candidates.append(("PATH", Path(which)))
    return candidates


def detect_nvcc() -> dict[str, Any]:
    candidates = nvcc_candidates()
    for source, candidate in candidates:
        if candidate.exists():
            result = run_command([str(candidate), "--version"])
            version_text = result.get("stdout") or result.get("stderr") or ""
            version_line = None
            for line in version_text.splitlines():
                if "release " in line or "Cuda compilation tools" in line:
                    version_line = line.strip()
                    break
            return {
                "found": True,
                "path": str(candidate),
                "path_source": source,
                "candidates": [{"source": item_source, "path": str(item_path)} for item_source, item_path in candidates],
                "version_line": version_line,
                "raw_output": version_text,
            }

    return {
        "found": False,
        "candidates": [{"source": item_source, "path": str(item_path)} for item_source, item_path in candidates],
    }


def detect_visual_studio() -> dict[str, Any]:
    if os.name != "nt":
        return {"found": False}

    base = Path(r"C:\Program Files\Microsoft Visual Studio")
    editions = ("Community", "Professional", "Enterprise", "BuildTools")
    years = ("2022", "2019")

    for year in years:
        for edition in editions:
            root = base / year / edition
            vcvars = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            msvc_root = root / "VC" / "Tools" / "MSVC"
            if not vcvars.exists() or not msvc_root.exists():
                continue

            toolsets = sorted([p for p in msvc_root.iterdir() if p.is_dir()], reverse=True)
            latest_toolset = toolsets[0] if toolsets else None
            cl_path = latest_toolset / "bin" / "Hostx64" / "x64" / "cl.exe" if latest_toolset else None

            data: dict[str, Any] = {
                "found": True,
                "year": year,
                "edition": edition,
                "vcvars64": str(vcvars),
                "msvc_root": str(msvc_root),
                "toolset_dir": latest_toolset.name if latest_toolset else None,
                "cl_path": str(cl_path) if cl_path and cl_path.exists() else None,
            }

            if cl_path and cl_path.exists():
                version_info = powershell_version_info(cl_path)
                if version_info:
                    data["cl_version_info"] = version_info
            return data

    return {"found": False}


def read_git_head(repo_path: Path) -> str | None:
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        return None

    if git_dir.is_file():
        return None

    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head

    ref = head[5:]
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()

    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("^") or not line.strip():
                continue
            sha, name = line.split(" ", 1)
            if name.strip() == ref:
                return sha.strip()
    return None


def detect_upstream_source(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    data: dict[str, Any] = {
        "path": str(source_dir),
        "exists": source_dir.exists(),
    }
    if not source_dir.exists():
        return data

    setup_path = source_dir / "setup.py"
    data["setup_py_exists"] = setup_path.exists()
    data["git_head"] = read_git_head(source_dir)
    return data


def collect(source_dir: Path) -> dict[str, Any]:
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "os": detect_os(),
        "python": detect_python(),
        "torch": detect_torch(),
        "flash_attn": detect_module("flash_attn"),
        "ninja": detect_ninja(),
        "nvcc": detect_nvcc(),
        "visual_studio": detect_visual_studio(),
        "upstream_source": detect_upstream_source(source_dir),
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a rebuild/debug environment fingerprint for this FlashAttention Windows repo.")
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional output path for the collected JSON report.",
    )
    parser.add_argument(
        "--flash-attn-src",
        type=Path,
        default=REPO_ROOT / "third_party" / "flash-attention-for-windows",
        help="Path to the upstream flash-attention-for-windows source tree.",
    )
    args = parser.parse_args()

    data = collect(args.flash_attn_src)
    rendered = json.dumps(data, indent=2)
    print(rendered)

    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
