# 🦙 👀 ⛑ DeepDR-LLM: Integrated Image-based Deep Learning and Language Models for Primary Diabetes Care

**DeepDR-LLM** offers a holistic approach to primary diabetes care by combining image-based deep learning with advanced language models. This repository includes code for utilizing the Vision Transformer (ViT) for image analysis, alongside fine-tuned LLaMA models to produce detailed management suggestions for patients with diabetes. Here, we employ the LLaMA-7B model as the foundational language model.

## Contents
1. Requirements
2. Environment Setup
   * Linux System
3. Dataset Preparation
4. Model Training and Evaluation

## Requirements
This software is compatible with a **Linux** operating system, specifically [**Ubuntu 20.04**](https://ubuntu.com/download/desktop) (compatibility with other versions has not been tested), and requires [**Python 3.9**](https://www.python.org). It necessitates **64GB of RAM** and **1TB of disk storage**. Performance benchmarks are based on an **Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz** and an **NVIDIA A100 GPU**.

The following **Python packages** are required, which are also listed in `requirements.txt`:
```
numpy>=1.25.0
datasets>=2.13.1
deepspeed>=0.10.0
huggingface-hub>=0.15.1
sentencepiece>=0.1.97
tokenizers>=0.13.1
torch>=2.0.1
transformers>=4.28.1
```

### Linux System
#### Step 1: Download the Project
1. Open the terminal, or press Ctrl+Alt+F1 to access the command line interface.
2. Clone this repository to your home directory.
```
git clone https://github.com/DeepPros/DeepDR-LLM.git
```
3. Navigate to the cloned repository's directory.
```
cd DeepDR-LLM
```
#### Step 2: Prepare the Environment and Execute the Code

1. Install the required Python packages.

```
python3 -m pip install --user -r requirements.txt
```
**Supported Image File Formats**
JPEG, PNG, and TIFF file formats are supported and have been tested. Other formats compatible with OpenCV should also work. The input image must be a 3-channel color fundus image with the shorter side of the resolution being greater than 448 pixels.

## Modules

### Module 1: Language Model (LLaMA) Integration

Module 1 leverages the LLaMA model to generate comprehensive diagnostic and treatment recommendations, designed for easy integration with outputs from Module 2.

1. **Dataset Preparation**
   * For training: 
   Ensure your dataset is formatted as shown in `DeepDR-LLM/Module1/Minimum/train_set/train_set.json`. Sample format: [{"instruction":"...","input":"...","output":"..."}]. 
   * For validation:
   Format your dataset according to the structure shown in `DeepDR-LLM/Module1/Minimum/valid_set.json`. Sample format: [{"instruction":"...","input":"...","output":"..."}]. 

2. **Training**
   - **Note**: Make sure `llama-7b` model weights are downloaded from `https://huggingface.co/huggyllama/llama-7b` and saved in `DeepDR-LLM/Module1/llama-7b-weights`.
   - Run `DeepDR-LLM/Module1/scripts/run_train.sh` to start training. 
   - Please review the settings in `run_train.sh`, particularly the `paths` configuration.

3. **Inference**

   See `DeepDR-LLM/Module1/scripts/inference.py` for guidance. Be sure to configure necessary arguments properly. The input format should match that in `DeepDR-LLM/Module1/Minimum/train_set/train_set.json`.

### Module 2: Image Prediction & Analysis

Module 2 is focused on analyzing and predicting outcomes based on fundus images.

1. **Dataset Preparation**

   Includes tasks for classification and segmentation. Datasets for both are compiled using .txt files, where each line corresponds to an image. For classification, the format is "imagepath classindex". For segmentation, it is "imagepath maskpath", with segmentation labels formatted as [C,H,W], where C includes the background category.

2. **Training**

   - For classification models, use `DeepDR-LLM/Module2/train_cla.py`.
   - For segmentation models, use `DeepDR-LLM/Module2/train_seg.py`.
   - **Note

**: Obtain pretrained `vit-base` model weights from ImageNet before training (https://download.pytorch.org/models/vit_b_16-c867db91.pth).

3. **Inference**

   Apply `DeepDR-LLM/Module2/test.py` for evaluation, ensuring trained models are accurately loaded. Outputs will be stored as specified.

### Integrated Workflow of DeepDR-LLM from Module 1 and Module 2
* Starting point: A fundus image is obtained from a standard or portable imaging device, along with aligned clinical metadata, following the example structure in `DeepDR-LLM/Module1/Minimum/train_set/train_set.json`.
#### Step 1: Submit the fundus image to Module 1 (using the 'test.py' script)
   - Predict the quality of the fundus image, DR grade, DME grade, and the presence of retinal lesions.
#### Step 2: Convert the clinical metadata into JSON format
   - Example: {Sex: Female; Age: 47; BMI: 22.13 kg/m^2;....}
#### Step 3: Combine the clinical metadata with the results from Module 1
   - Example: {Sex: Female; Age: 47; BMI: 22.13 kg/m^2;....; Fundus Image Quality: Gradable; DR Grade: 0; DME Grade: 0; Presence of Retinal Lesions: No microaneurysms, no cotton-wool spots, no hard exudates, no hemorrhages.}
#### Step 4: Input the combined data from Step 3 into Module 2 to obtain the final output
