# scCLIP
single-cell CLIP model




# Requirements

other tools (tabix)

tabix can be downloaded via [htslib]{https://www.htslib.org/download/}
```bash
cd htslib
make configure
make
#in the .bashrc 
export PATH=/scratch/project_465001820/htslib-1.21:$PATH
```

Python packages required

```bash
pip install snapatac2
pip install scanpy
pip install --user scikit-misc
```
flashattention is required to accelerate training and inference with the accuracy kept.
Before that, nvcc(CUDA compiler) should be detected in your device by installing the nvcc via
```bash
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
#check the installation of nvcc
nvcc --version
```
When compilation is ready, let's install the flash-attention
```bash
pip install flash-attn
```

# Updates

2025/2/26 JunWang upload the codes of all available packages for sequence generation
2025/3/18 JunWang upload the data preprocessing and model architecture to the repository
