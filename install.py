import subprocess
import sys
import platform

def install_packages(packages):
    """Instala una lista de paquetes"""
    if isinstance(packages, str):
        packages = [packages]
    print(f"[INFO] Installing: {packages}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

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

    # Install PyTorch - VERSIÓN CORREGIDA
    if gpu_detected:
        print("[INFO] Installing PyTorch with CUDA 11.8 support...")
        install_packages([
            "torch", 
            "torchvision", 
            "torchaudio",
            "--index-url", 
            "https://download.pytorch.org/whl/cu118"
        ])
        torch_type = "GPU (CUDA 11.8)"
    else:
        print("[INFO] Installing PyTorch for CPU...")
        install_packages([
            "torch", 
            "torchvision", 
            "torchaudio",
            "--index-url", 
            "https://download.pytorch.org/whl/cpu"
        ])
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
        install_packages(dep)

    # Verificar que PyTorch se instaló correctamente
    try:
        import torch
        print(f"\n[SUCCESS] PyTorch version: {torch.__version__}")
        if gpu_detected:
            print(f"[SUCCESS] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[SUCCESS] GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\n[ERROR] PyTorch no se instaló correctamente")

    print("\n[INFO] ===== INSTALLATION SUMMARY =====")
    print(f"[INFO] Operating system: {os_name}")
    print(f"[INFO] NVIDIA GPU detected: {gpu_detected}")
    print(f"[INFO] PyTorch installed: {torch_type}")
    print("[INFO] Additional dependencies installed successfully.")
    print("[INFO] Installation completed.\n")

if __name__ == "__main__":
    main()