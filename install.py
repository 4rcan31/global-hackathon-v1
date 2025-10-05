import subprocess
import sys
import platform

def install(package):
    print(f"[INFO] Installing: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())

def has_nvidia_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def main():
    print("\n[INFO] Starting dependency installation...\n")

    os_name = platform.system().lower()
    print(f"[INFO] Detected operating system: {os_name}")

    gpu_detected = has_nvidia_gpu()
    print(f"[INFO] NVIDIA GPU detected: {gpu_detected}")

    # Install PyTorch
    if gpu_detected:
        print("[INFO] Installing PyTorch with CUDA support...")
        install("--index-url https://download.pytorch.org/whl/cu121 torch torchvision")
        torch_type = "GPU (CUDA 12.1)"
    else:
        print("[INFO] Installing PyTorch for CPU...")
        install("--index-url https://download.pytorch.org/whl/cpu torch torchvision")
        torch_type = "CPU"


    print("\n[INFO] Installing additional dependencies...\n")
    dependencies = [
        "opencv-python",
        "numpy",
        "pillow",
        "ftfy",
        "regex",
        "tqdm",
        "ultralytics",
        "git+https://github.com/openai/CLIP.git"
    ]
    for dep in dependencies:
        install(dep)

    print("\n[INFO] ===== INSTALLATION SUMMARY =====")
    print(f"[INFO] Operating system: {os_name}")
    print(f"[INFO] NVIDIA GPU detected: {gpu_detected}")
    print(f"[INFO] PyTorch installed: {torch_type}")
    print("[INFO] Additional dependencies installed successfully.")
    print("[INFO] Installation completed.\n")

if __name__ == "__main__":
    main()


# ================= ESCENARIOS POSIBLES =================
# Escenario 1: Equipo sin GPU → PyTorch CPU instalado
# Escenario 2: Equipo con GPU pero PyTorch CPU instalado inicialmente
#             → PyTorch CUDA se instalará automáticamente, sobrescribiendo CPU
# Escenario 3: Equipo con GPU y PyTorch CUDA ya instalado
#             → Script detecta GPU y confirma instalación CUDA existente, no hay cambios
