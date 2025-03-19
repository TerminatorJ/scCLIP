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

FlashAttention is required to accelerate training and inference while maintaining accuracy.  
Before that, nvcc(CUDA compiler) should be detected in your device by installing the nvcc via
```bash
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
#check the installation of nvcc
nvcc --version
```
When compilation is ready, let's install the flash-attention  

To get started with the triton backend for **AMD**, follow the steps below.
```bash
pip install triton
```
Then install the FlashAttention(2.X) from the github
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
cd flash-attention
python setup.py install
```
Finally, test whether it works normally.
```bash
pytest tests/test_flash_attn.py
```

Alternatively, if you are using **A100**, please easily run the following code to install FlashAttention(2.X)
```bash
pip install flash-attn --no-build-isolation
```

We implement the FlashAttention(2.x) in our code, which is completely reweited and 2x faster than FlashAttention(1.x).

# Updates

2025/2/26 JunWang upload the codes of all available packages for sequence generation  
2025/3/18 JunWang upload the data preprocessing and model architecture to the repository
2025/3/18 JunWang upload the full model architechture, setup the demo model to be trained and optimized
