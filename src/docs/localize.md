# PROJECT DIRECTORY MAP (DR Grading and XAI)

*Note: All active python scripts are kept in the root directory to prevent Python import routing issues. Use this map to navigate the project's logical workflow.*

## 1. PREPROCESSING AND DATA PREPARATION
Scripts used for downloading, cleaning, and preparing data for training and clinical grading.

| File Name | Description |
| :--- | :--- |
| `preprocessing.py` | Contains core OpenCV functions (Black-border cropping, CLAHE enhancement). |
| `augmentation.py` | Contains Albumentations pipelines for Train (Heavy TTA) and Valid/Test sets. |
| `create_folds.py` | Applies Stratified Group K-Fold to split the APTOS dataset. |
| `sample_10_maples.py` | Extracts 10 representative images and builds Master Masks from MAPLES-DR. |
| `prepare_blind_grading.py` | Extracts 198 MAPLES-DR images and generates a CSV template for independent clinical grading. |
| `verifydataset.py` | Sanity check script to ensure no images are missing or corrupted. |
| `audit_labels.py` | Audits the label mapping from MESSIDOR (R0-R4) to APTOS grades (0-4). |

## 2. MODEL TRAINING AND TESTING
Scripts responsible for building the architecture and fine-tuning the model across datasets.

| File Name | Description |
| :--- | :--- |
| `model.py` | Defines the APTOSModel architecture, CBAM Attention Module, and Smooth L1 Loss. |
| `rounder.py` | Implements OptimizedRounder (Nelder-Mead) to map continuous regression outputs to discrete grades (0-4). |
| `train_stage1.py` | Pre-training script using 35,000 images from EyePACS 2015 dataset. |
| `train_stage2.py` | Fine-tuning script on APTOS 2019 using TTA and early stopping. |
| `train_stage2_messidor.py` | Fine-tuning script specialized for the MESSIDOR dataset. |
| `train_stage3_attention.py` | Attention-Guided fine-tuning (using MAPLES-DR Train masks to guide the CBAM loss). |

## 3. EXPLAINABLE AI AND CCEM
The core contribution scripts. These fuse the explanations and benchmark them against Expert Ground Truths.
XAI methods: explanation
result: ../scripts/XAI_10ex_run (Contains 10 examples from each grade from maples-dr lession mark + messidor images)

## 4. ARCHIVE
The `Archive` folder contains deprecated scripts (e.g., LIME implementation, standalone RISE, old fusion logic, and legacy evaluation metrics) that were previously used during the methodology development phase. They are kept for historical reference and ablation studies.