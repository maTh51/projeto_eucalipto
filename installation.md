# Installation and Setup Guide

This guide describes how to install and configure the **Eucalyptus Forest Point Cloud Pipeline** on a new computer. 

Since the project uses Unix-specific Python calls (such as `os.sync()`), hardcoded temporary staging directories (`/tmp`), and shell script wrappers to invoke Docker containers, **running natively on Windows is not supported. Running via WSL 2 (Windows Subsystem for Linux) is highly recommended and required.**

---

## Environment Prerequisites

Ensure you have the following on the host machine:
- **OS**: Windows 10/11 (Build 19041 or higher) or a modern Linux distribution (e.g. Ubuntu 22.04).
- **GPU**: NVIDIA GPU with latest drivers installed on the host.
- **Disk Space**: At least 30-50 GB of free space (for Docker images, datasets, and pre-trained weights).

---

## Step 1: Install WSL 2 & Configure NVIDIA Drivers (Windows Only)

WSL 2 allows you to run a native Linux environment directly inside Windows. Crucially, **NVIDIA GPU drivers are installed once on the Windows host and automatically inherited by WSL 2**.

1. **Install NVIDIA Drivers on Windows Host:**
   - Download and install the latest game or studio driver for your GeForce, RTX, or Quadro card using the **NVIDIA App**, **GeForce Experience**, or manually from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx).
   - *Do not install any NVIDIA GPU drivers inside your WSL 2 Ubuntu terminal. Doing so will break the WSL driver bridge.*

2. **Install WSL 2:**
   - Open PowerShell or Windows Command Prompt as Administrator and run:
     ```powershell
     wsl --install -d Ubuntu
     ```
   - Restart your computer if prompted by the installer.
   - Upon restart, complete the Ubuntu username/password setup in the terminal window that pops up.

3. **Verify GPU Bridge in WSL 2:**
   - Open your WSL 2 terminal and run:
     ```bash
     nvidia-smi
     ```
   - If this command prints your GPU model and driver version, the GPU driver is successfully bridged from Windows.

For more details, see the [Microsoft WSL Installation Documentation](https://learn.microsoft.com/en-us/windows/wsl/install).

---

## Step 2: Install Docker Desktop & Enable WSL 2 integration

The isolation (TreeISO, ForestFormer3D) and semantic segmentation (Leaf-Wood) pipelines rely heavily on Docker containers.

1. Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2. During installation, make sure the **WSL 2 backend** option is checked.
3. Open Docker Desktop settings:
   - Go to **General** and check **Use the WSL 2 based engine**.
   - Go to **Resources** > **WSL integration** and toggle **Ubuntu** (or your default distro) to **On**.
   - Click **Apply & restart**.
4. Test that Docker is responsive and has GPU access in WSL 2:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
   *(This should output a CUDA device table matching your physical GPU).*

For more details, see the [Docker WSL 2 Integration Guide](https://docs.docker.com/desktop/wsl/).

---

## Step 3: Install Core Linux Dependencies

Inside your WSL 2 terminal (or native Linux shell), install git, Python package management tools, and zip utilities:

```bash
sudo apt update && sudo apt install -y \
    git \
    python3-pip \
    python3-venv \
    build-essential \
    wget \
    unzip
```

---

## Step 4: Clone the Repository and Submodules

Clone this repository and pull all third-party submodules recursively:

```bash
git clone <repository_url> projeto_eucalipto
cd projeto_eucalipto

# Initialize external dependencies (FF3D, TreeISO, leafwood-segmentation, rayextract)
git submodule update --init --recursive
```

---

## Step 5: Run the Project Setup Script

The [setup.sh](file:///home/grad/ccomp/18/matheuspimenta/euc/projeto_eucalipto/setup.sh) script automates the creation of the Python virtual environment, updates packaging tools, installs host dependencies in editable mode, and downloads pretrained weights (~1.5 GB total) from Zenodo:

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup
./setup.sh
```

### What `setup.sh` does:
1. **Updates git submodules** to ensure third-party tools are present.
2. **Creates a local Python virtual environment** under `.venv` and installs the requirements listed in [requirements.txt](file:///home/grad/ccomp/18/matheuspimenta/euc/projeto_eucalipto/requirements.txt) and [pyproject.toml](file:///home/grad/ccomp/18/matheuspimenta/euc/projeto_eucalipto/pyproject.toml).
3. **Creates target folders** for the model weights.
4. **Downloads and extracts the ForestFormer3D weights** (`epoch_3000_fix.pth`) to `third_party/FF3D_inference/ff3d_forestsens/work_dirs/clean_forestformer/`.
5. **Downloads and extracts Leaf-Wood weights** (`weights_randlanet.pth`, etc.) to `third_party/leaf-wood-segmentation-with-deep-learning/model_weights/`.

---

## Step 6: Verify and Run the Pipeline

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Test your setup** using the default ForestFormer3D configuration file:
   - Edit the input and output paths in [configs/ff3d_full.yaml](file:///home/grad/ccomp/18/matheuspimenta/euc/projeto_eucalipto/configs/ff3d_full.yaml) to point to your local LAS/LAZ point cloud file.
   - Run the execution script:
     ```bash
     python run_pipeline.py configs/ff3d_full.yaml
     ```
3. **Verify the outputs**:
   - The results will be stored in your configured output directory (e.g. `results/canoa/ff3d_full`).
   - Check for the existence of `classified_cloud.laz`, `metrics.csv`, and `run_manifest.json`.

---

## Troubleshooting & WSL 2 Performance Tips

### Docker Compose out-of-memory or out-of-CPU errors
WSL 2 defaults to consuming up to 50% of your system RAM. If Docker containers crash during feature extraction or deep learning inference, you can allocate more resources by editing your Windows WSL configuration.

1. In Windows, press `Win + R`, type `%USERPROFILE%`, and press Enter.
2. Create or edit a file named `.wslconfig` and add:
   ```ini
   [wsl2]
   memory=16GB    # Adjust to 75% of your host RAM
   processors=8   # Adjust to the number of physical cores
   ```
3. Restart WSL in powershell: `wsl --shutdown`

### Pyransac3d / Open3D compilation issues
Host Python dependencies are installed using pre-compiled wheels. If you receive compilation errors during `pip install`, ensure you have installed the `build-essential` package inside WSL 2 (see Step 3).
