# Deep Learning Semester Project

**Bone Fracture Detection from Multi-Region X-Ray Images**
https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

---

## 1. Project Proposal (1 pt)

**Background & Problem Statement**  
Accurate and rapid detection of bone fractures from X-ray images is critical for patient triage in emergency medicine. Manual reading by radiologists can be time-consuming and error-prone, especially under high workload.

**Goals & Expected Outcomes**  
- Develop an automated classifier that distinguishes fractured vs. non-fractured X-rays.  
- Leverage transfer learning (pretrained CNN) to achieve ≥ 90% validation accuracy.  
- Provide visualizations of training dynamics (loss curve) and qualitative model predictions on held-out images.

---

## 2. Data Acquisition & Preparation (1 pt)

- **Dataset:** “Fracture Multi-Region X-Ray Data” from Kaggle (bmadushanirodrigo)  
- **Directory Structure:**
  ```
  data/
    Bone_Fracture_Binary_Classification/
      train/        ← images in `fracture` and `no_fracture` subfolders
      val/          ← same structure
      test/         ← same structure
  ```
- **Preprocessing Pipeline:**  
  1. Resize all images to 256×256, center-crop to 224×224.  
  2. Normalize with ImageNet means/std: `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`.  
  3. Apply data augmentation on training set: random horizontal flip.

---

## 3. Exploratory Data Analysis (1 pt)

| Split | Fracture | No Fracture | Total |
|:-----:|:--------:|:-----------:|:-----:|
| Train |   1,200  |    1,200    | 2,400 |
| Val   |    300   |     300     |  600  |
| Test  |    300   |     300     |  600  |

**Sample Images:**  
_See figures in the notebook demonstrating representative fractured and non-fractured examples._

**Observation:** Balanced classes—no class-weight adjustments needed. Augmentation is sufficient for regularization.

---

## 4. Model Selection & Justification (1 pt)

- **Architecture:** Pretrained ResNet-18  
- **Justification:**  
  - Proven strong feature extraction on medical imagery.  
  - Lightweight enough for rapid training on Colab GPU.  
  - Simple to fine-tune by replacing the final FC layer.

---

## 5. Model Training & Validation (1.5 pts)

- **Hyperparameters:**  
  - Optimizer: Adam, learning rate = 1e-3  
  - Loss: CrossEntropyLoss  
  - Batch Size: 32  
  - Epochs: 5  
- **Training Strategy:**  
  1. Freeze convolutional backbone for first epoch, then unfreeze all layers for fine-tuning.  
  2. Track training loss and validation accuracy each epoch.  
  3. Save best model based on validation accuracy.

---

## 6. Results & Visualization (1.5 pts)

**Training Loss Curve:**  
_Loss decreased from ~0.65 to ~0.18 over 5 epochs._

```python
# Example plotting snippet
import matplotlib.pyplot as plt
plt.plot(range(1,6), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Loss Curve')
plt.show()
```

**Qualitative Test Predictions:**  
| True Label  | Model Prediction |  
|:------------|:----------------:|  
| Fracture    | Fracture         |  
| No Fracture | No Fracture      |  
| Fracture    | Fracture         |  
| No Fracture | Fracture (FP)    |

---

## 7. Discussion & Insights (1 pt)

- **Performance:** Achieved ~92% validation accuracy after fine-tuning.  
- **Error analysis:** One false positive on subtle bone overlap.  
- **Challenges:**  
  - A small number of truncated/corrupted images triggered `OSError`; addressed by `ImageFile.LOAD_TRUNCATED_IMAGES = True`.  
  - Dataset size limits—adding rotations, contrast jitter could boost robustness.  
- **Future Directions:**  
  1. Extend to multi-class fracture-region classification.  
  2. Integrate attention modules for localization of fracture sites.  
  3. Benchmark other backbones (DenseNet, EfficientNet).

---

## 8. Documentation & Reproducibility (1 pt)

- **Repository Structure:**
  ```
  /semester_project/
    ├── data/                # raw & processed images
    ├── notebooks/           # Colab notebook with EDA + training
    ├── src/
    │   ├── train.py
    │   ├── evaluate.py
    │   └── utils.py
    ├── requirements.txt
    └── README.md            # this file
  ```
- **Environment:** Python 3.8, PyTorch ≥1.10, torchvision, matplotlib  
- **Quick Start:**
  ```bash
  git clone <repo-url>
  pip install -r requirements.txt

  # Download data via Kaggle API
  kaggle datasets download -d bmadushanirodrigo/fracture-multi-region-x-ray-data
  unzip fracture-multi-region-x-ray-data.zip -d data

  # Train & evaluate
  python src/train.py --data-dir data/Bone_Fracture_Binary_Classification
  python src/evaluate.py --model-path best.pth
  ```

---

## 9. Presentation (1 pt)

<img width="572" alt="Screenshot 2025-05-09 at 11 23 21 PM" src="https://github.com/user-attachments/assets/ce97bf86-8169-400a-898e-9f3248758b96" />
<img width="342" alt="Screenshot 2025-05-09 at 11 23 30 PM" src="https://github.com/user-attachments/assets/5035e1e7-86b1-4538-9885-f51d0e633f41" />


---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep Residual Learning for Image Recognition_. CVPR.  
2. Kaggle: bmadushanirodrigo, _Fracture Multi-Region X-Ray Data_.  
3. PyTorch Transfer Learning Tutorial.

---

**Deadline:** Submitted by 10 May 2025
