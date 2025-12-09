# 🚀 RAFT Optical Flow - MacBook Setup Guide
### Recurrent All-Pairs Field Transforms for Optical Flow
**Implemented by Rohan Tikotekar** | UC Riverside MS CS Student

<div align="center">

![RAFT Demo](RAFT.png)

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.6+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD--3-green.svg)](LICENSE)

*Original Paper: [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf) - ECCV 2020*

*Implementation video recording by Rohan Tikotekar: https://drive.google.com/file/d/1DWUD1YhBWhCMHFeTFREpHL9-dZ0yiWKW/view?usp=sharing
</div>

---


## 📋 Table of Contents
- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
- [Running Demos](#-running-demos)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Credits](#-credits)

---

## 🎯 Overview

This repository contains a **MacBook-optimized implementation** of RAFT (Recurrent All-Pairs Field Transforms), a state-of-the-art optical flow estimation model. The original implementation required CUDA GPUs, but this version has been adapted to run on **Intel and Apple Silicon Macs using CPU**.

### What is Optical Flow?
Optical flow estimates the motion of objects between consecutive frames in a video sequence. RAFT achieves this with high accuracy using recurrent neural networks.

---

## ⚡ Quick Start
```bash
# 1. Activate environment
source ~/miniconda3/bin/activate raft

# 2. Navigate to project
cd ~/Documents/RAFT

# 3. Run demo with saved outputs
python demo_save_output.py --model=models/raft-things.pth --path=demo-frames

# 4. View results
open output_flows
```

---

## 💻 System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | macOS (Intel or Apple Silicon) |
| **Python** | 3.8 |
| **RAM** | 8GB minimum (16GB recommended) |
| **Storage** | 2GB free space |
| **Tools** | Git, Miniconda/Anaconda |

---

## 🛠️ Installation Guide

### Step 1: Install Miniconda
```bash
# Download Miniconda for Intel Mac
cd ~/Downloads
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Install
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3

# Initialize
~/miniconda3/bin/conda init zsh
```

**Close and reopen your terminal after this step.**

### Step 2: Clone Repository
```bash
cd ~/Documents
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
```

### Step 3: Create Environment
```bash
# Create conda environment
conda create --name raft python=3.8 -y

# Activate environment
conda activate raft

# Install PyTorch (CPU version for Mac)
conda install pytorch torchvision -c pytorch -y

# Install dependencies
conda install matplotlib tensorboard scipy opencv -c conda-forge -y
```

### Step 4: Download Pre-trained Models
```bash
# Method 1: Using the provided script
chmod +x download_models.sh
./download_models.sh

# Method 2: Manual download (if script fails)
pip install gdown
mkdir -p models
cd models
gdown 1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT --folder
cd ..
```

### Step 5: Create Mac-Compatible Demo Script
```bash
cat > demo_save_output.py << 'EOF'
import sys
sys.path.append('core')
import argparse, os, cv2, glob, numpy as np, torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module.to(DEVICE).eval()
    
    os.makedirs('output_flows', exist_ok=True)
    
    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path, '*.png')) + 
                       glob.glob(os.path.join(args.path, '*.jpg')))
        
        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1, image2 = load_image(imfile1), load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            flo_viz = flow_viz.flow_to_image(flo)
            
            output_file = f'output_flows/flow_{idx:04d}.png'
            cv2.imwrite(output_file, flo_viz[:, :, [2,1,0]])
            
            np.save(f'output_flows/flow_{idx:04d}.npy', flo)
            
            print(f'✓ Saved: {output_file}')
            print(f'  {os.path.basename(imfile1)} → {os.path.basename(imfile2)}')
    
    print(f'\n🎉 All results saved to: output_flows/')
    print(f'📊 Total: {len(images)-1} optical flow visualizations')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Path to model checkpoint")
    parser.add_argument('--path', help="Path to input frames")
    parser.add_argument('--small', action='store_true', help='Use small model')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    demo(parser.parse_args())
EOF
```

---

## 🎬 Running Demos

### Basic Demo (with saved outputs)
```bash
python demo_save_output.py --model=models/raft-things.pth --path=demo-frames
```

### View Results
```bash
# Open output folder
open output_flows

# View specific result
open output_flows/flow_0000.png

# List all outputs
ls -lh output_flows/
```

### Using Your Own Images
```bash
# Create a folder with your sequential frames
mkdir my_frames
# Add your images (frame_001.png, frame_002.png, etc.)

# Run RAFT
python demo_save_output.py --model=models/raft-things.pth --path=my_frames
```

---

## 📁 Project Structure
```
RAFT/
├── 📂 core/                    # Core RAFT implementation
│   ├── raft.py                # Main model architecture
│   ├── update.py              # GRU update module
│   └── utils/                 # Utility functions
├── 📂 demo-frames/            # Sample input frames
├── 📂 models/                 # Pre-trained model weights
│   └── raft-things.pth       # Main model checkpoint
├── 📂 output_flows/           # Generated optical flow visualizations
├── 📜 demo_save_output.py     # Mac-compatible demo script
├── 📜 download_models.sh      # Model download script
└── 📜 README.md               # This file
```

---

## 🔧 Troubleshooting

### Issue: `conda: command not found`

**Solution:**
```bash
source ~/miniconda3/bin/activate raft
```

Or permanently fix by adding to shell:
```bash
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Issue: `.zshrc` parse error

**Solution:**
```bash
cat > ~/.zshrc << 'EOF'
# Conda initialization
__conda_setup="$('/Users/EndUser/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/EndUser/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/EndUser/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/EndUser/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
EOF
```

### Issue: Model download fails

**Solution:**
```bash
pip install gdown
mkdir -p models
cd models
gdown 1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT --folder
cd ..
```

### Issue: Out of memory on Mac

**Solution:** Process fewer frames at once or reduce image resolution:
```python
# In demo_save_output.py, add downsampling:
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
```

---

## 📊 Performance Notes

| Setup | Inference Speed | Memory Usage |
|-------|----------------|--------------|
| **Intel Mac (CPU)** | ~5-10 sec/frame pair | ~2-4 GB RAM |
| **Apple Silicon (CPU)** | ~3-7 sec/frame pair | ~2-4 GB RAM |
| **Original CUDA GPU** | ~0.1 sec/frame pair | ~4-8 GB VRAM |

*Note: MacBook performance is significantly slower than GPU but sufficient for research and small-scale testing.*

---

## 🎓 Educational Use

This implementation is part of my graduate coursework at **UC Riverside** focusing on:
- Computer Vision
- Deep Learning for Autonomous Systems
- Optical Flow Estimation
- Model Deployment on Resource-Constrained Devices

---

## 📚 Additional Resources

- 📄 [Original RAFT Paper](https://arxiv.org/pdf/2003.12039.pdf)
- 🔗 [Original Repository](https://github.com/princeton-vl/RAFT)
- 📖 [PyTorch Documentation](https://pytorch.org/docs/)
- 🎥 [ECCV 2020 Presentation](https://www.youtube.com/watch?v=BMBCFzkMBrI)

---

## 🏆 Credits

### Original Authors
- **Zachary Teed** - Princeton Vision & Learning Lab
- **Jia Deng** - Princeton University

### MacBook Implementation
- **Rohan Tikotekar** - UC Riverside
  - MS Computer Science (AI/ML Track)
  - Email: rtiko001@ucr.edu
  - LinkedIn: [linkedin.com/in/rohantikotekar](https://linkedin.com/in/rohantikotekar)

---

## 📝 License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

Special thanks to:
- Princeton Vision & Learning Lab for the original RAFT implementation
- PyTorch team for the deep learning framework
- UC Riverside TASL Lab for computational resources
- Dr. [Advisor Name] for project guidance

---

<div align="center">

### 🚀 Ready to estimate optical flow on your Mac!

**Made with ❤️ by Rohan Tikotekar**

[⭐ Star this repo](https://github.com/yourusername/RAFT) • [🐛 Report Bug](https://github.com/yourusername/RAFT/issues) • [💡 Request Feature](https://github.com/yourusername/RAFT/issues)

</div>
