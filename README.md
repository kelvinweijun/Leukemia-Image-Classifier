# Leukemia Subtype Classification Using Multi-Cell Images

## Abstract

Blood cancer remains one of the most pressing concerns globally, and its means of diagnosis is still a challenge to this day. Current image-based research on blood cancer cells primarily focuses on single- cell images often neglecting the spatial features between multiple cells. Our aim is to train models for multi-classification task to classify leukemia subtypes using the Acute Lymphoblastic Leukemia (ALL) dataset, which comprises a diverse collection of 3,256 multi-cell blood smear images classed according to their maturity subtypes. In this project, we also proposed a new Convolutional Neural Network (CNN) architecture evaluated multiple machine learning models, including SVM, XGBoost, VGG16, DenseNet- 201, and ConvNeXt. We focused on optimizing accuracy while considering the misclassification costs associated with different leukemia stages, and found that ConvNeXt and DenseNet-201 pretrained model combined with our proposed architecture performs the best, with DenseNet-201 topping in performance across accuracy and having minimal misclassification costs.

##  Dataset

The models in this study will be using the archived dataset of Acute Lymphoblastic Leukemia (ALL) blood cancer, which provides a large collection of 3,242 peripheral blood smear (PBS) images, organized into two subfolders where one contains the original images captured, and the other one pre-segmented. Within each of these two subfolders contains folders which hold images based on developmental subtypes: Benign, Early Pre-B, Pre-B, and Pro-B ALL. The dataset was prepared in the bone marrow laboratory of Taleqani Hospital in Tehran, Iran.

---

## Features

- Multi-cell spatial feature analysis
- CNN-based custom architecture
- Transfer learning with pretrained models (VGG16, DenseNet-201, ConvNeXt)
- Evaluation using classification accuracy and misclassification cost metrics

---

## Model Architectures

**Support Vector Machine (SVM)**
**XGBoost**
**VGG16**
**DenseNet-201**
**ConvNeXt**

---

## Results

| Model         | Accuracy | Misclassification Cost | Notes                            |
|---------------|----------|-------------------------|----------------------------------|
| SVM           | ~        | ~                       | Baseline traditional model       |
| XGBoost       | ~        | ~                       | Tree-based ensemble              |
| VGG16         | ~        | ~                       | Pretrained CNN                   |
| DenseNet-201  | ✅ Best  | ✅ Lowest                | Top performer                    |
| ConvNeXt      | ✅ High  | ✅ Low                   | Strong contender                 |

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
    To train the model, open any of the IPYNB notebooks and run them with a suitable software. Preferably Kaggle, Colab or Jupyter notebook.

4. Evaluate performance:
    To evaluate performance, access this link: 

---

## 🤝 Acknowledgements

This work was made possible by the ALL image dataset and the support of deep learning libraries such as Keras and Computer Vision models.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
