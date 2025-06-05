# Leukemia Subtype Classification Using Multi-Cell Images

## Abstract

Blood cancer remains one of the most pressing global health concerns, with diagnosis still posing significant challenges. Current image-based research on blood cancer cells primarily focuses on single-cell images, often overlooking spatial relationships between multiple cells.

This project aims to address that gap by training models for a multi-classification task to classify leukemia subtypes using the **Acute Lymphoblastic Leukemia (ALL)** dataset. The dataset comprises **3,256** multi-cell blood smear images categorized by their maturity subtypes.

We proposed a novel **Convolutional Neural Network (CNN)** architecture and evaluated multiple machine learning models, including:

- SVM
- XGBoost
- VGG16
- DenseNet-201
- ConvNeXt

Our focus was on optimizing classification **accuracy** while also minimizing **misclassification costs** across different leukemia stages.

Key findings:
- **ConvNeXt** and **DenseNet-201** pretrained models, especially when combined with our proposed CNN architecture, showed the best performance.
- **DenseNet-201** achieved the highest overall accuracy with minimal misclassification costs.

---

##  Dataset

- **Name**: Acute Lymphoblastic Leukemia (ALL)
- **Type**: Multi-cell blood smear images
- **Size**: 3,256 images
- **Classes**: Benign, Early, Pre, Pro (based on cell maturity)

---

## Features

- Multi-cell spatial feature analysis
- CNN-based custom architecture
- Transfer learning with pretrained models (VGG16, DenseNet-201, ConvNeXt)
- Evaluation using classification accuracy and misclassification cost metrics

---

## Model Architectures

- ✅ **Custom CNN**
- ✅ **Support Vector Machine (SVM)**
- ✅ **XGBoost**
- ✅ **VGG16**
- ✅ **DenseNet-201**
- ✅ **ConvNeXt**

---

## Results

| Model         | Accuracy | Misclassification Cost | Notes                            |
|---------------|----------|-------------------------|----------------------------------|
| SVM           | ~        | ~                       | Baseline traditional model       |
| XGBoost       | ~        | ~                       | Tree-based ensemble              |
| VGG16         | ~        | ~                       | Pretrained CNN                   |
| DenseNet-201  | ✅ Best  | ✅ Lowest                | Top performer                    |
| ConvNeXt      | ✅ High  | ✅ Low                   | Strong contender                 |
| Custom CNN    | ✅       | ✅                       | Proposed architecture used       |

(*Exact values to be added based on experiment logs*)

---

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/kelvinweijun/Leukemia-Image-Classifier.git
    cd Leukemia-Image-Classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:
    ```bash
    python train.py --model densenet201
    ```

4. Evaluate performance:
    ```bash
    python evaluate.py
    ```

---

## 📈 Future Work

- Incorporating attention mechanisms
- Leveraging segmentation masks
- Expansion to other types of blood cancers

---

## 🤝 Acknowledgements

This work was made possible by the ALL image dataset and the support of deep learning libraries such as TensorFlow and PyTorch.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
