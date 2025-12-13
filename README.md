# SegFormer Pipeline for 3D Ultrasound Muscle Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow)](https://huggingface.co/)
[![3D Slicer](https://img.shields.io/badge/Integration-3D%20Slicer-lightgrey)](https://www.slicer.org/)

## 📋 Overview

This repository contains the official implementation of the **SegFormer-based End-to-End Pipeline** developed for the Bachelor Thesis *"Schnelle 3D-Segmentierung in 3D Slicer mit einem SegFormer-basierten Pipeline-Ansatz"* at the **University of Basel (DSBG)**.

The pipeline automates the segmentation of the *M. vastus lateralis* in volumetric ultrasound data. It processes raw **NRRD** files, performs 2D inference using a fine-tuned **SegFormer** model (via Hugging Face), and reconstructs the segmentation into a 3D volume compatible with **3D Slicer**.

### Key Features
* **End-to-End Automation:** Converts `.nrrd` volumes to slice batches, runs inference, and reconstructs the 3D mask.
* **Hugging Face Integration:** Utilizes the robust `transformers` library for model weights and architecture.
* **Performance:** Achieves a Max IoU of **0.943**, significantly outperforming traditional CNN baselines (VGG, TransUNet).
* **Metadata Preservation:** Maintains correct voxel spacing and origin for seamless integration back into 3D Slicer.

---

## ⚙️ Installation

### Prerequisites
* Windows / Linux
* **NVIDIA GPU** (Highly recommended for inference speed) with CUDA installed.
* Anaconda or Miniconda

### Environment Setup
Create a new environment and install the required dependencies (PyTorch, Transformers, PyNRRD, Albumentations, etc.):

```bash
# 1. Create environment
conda create -n segformer python=3.10
conda activate segformer

# 2. Install PyTorch (Check [https://pytorch.org/](https://pytorch.org/) for your specific CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install pipeline dependencies
pip install transformers numpy pynrrd tifffile ipywidgets albumentations pytorch-lightning pillow

## 🚀 Usage

The core of the project is the Jupyter Notebook `script.ipynb`.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dgoncasimao/segformer.git](https://github.com/dgoncasimao/segformer.git)
    cd segformer
    ```

2.  **Start Jupyter:**
    ```bash
    jupyter notebook
    ```

3.  **Run the Pipeline:**
    Open `script.ipynb` and follow the interactive steps:
    * **Step 1:** Load dependencies.
    * **Step 2 (Hyperparameters):** Adjust `threshold` (default 0.5) and `crop_margin`. *Note: Correct tuning is essential to avoid noise artifacts.*
    * **Step 3 (Input):** Provide the path to your input volume (e.g., `data/volume.nrrd`).
    * **Step 4 (Inference):** The script executes slice-by-slice prediction using the SegFormer weights.
    * **Step 5 (Reconstruction):** The output mask is saved as `volume_segmentation.nrrd` in the same folder.

## ⚠️ Known Limitations

While the segmentation performance is high (IoU > 0.94), the current pipeline has specific constraints discussed in the thesis:

1.  **Reconstruction Incoherence (Z-Axis):** The pipeline operates on a 2D slice-by-slice basis. The final 3D volume is constructed by stacking these masks without 3D smoothing. This can result in "step-like" artifacts or slight inconsistencies between adjacent slices.
2.  **Proximal Slices:** The model may struggle at the very origin of the muscle (proximal area) due to low contrast and lack of 3D context in the inference step.
3.  **Preprocessing Sensitivity:** The quality of the output is highly dependent on the `threshold` and `crop_margin` settings. Incorrect parameters can lead to fragmented masks ("Garbage In, Garbage Out"), requiring manual adjustment for different ultrasound gain settings.

## 👥 Credits

* **Author:** Diego Gonçalves Simão
* **Supervisor:** Paul Ritsche
* **Institution:** University of Basel, Department of Sport, Exercise and Health (DSBG)

### Acknowledgments
* **Hugging Face:** We gratefully acknowledge the use of the [Hugging Face Transformers](https://huggingface.co/) library for the SegFormer implementation and pre-trained weights.
* **AI Assistance:** Parts of the code, documentation, and translation work were refined with the assistance of **ChatGPT** (OpenAI) and **Gemini** (Google).
