# Leukemia Subtype Classification Using Multi-Cell Images

## Abstract

Blood cancer remains one of the most pressing concerns globally, and its means of diagnosis is still a challenge to this day. Current image-based research on blood cancer cells primarily focuses on single- cell images often neglecting the spatial features between multiple cells [1]. Our aim is to train models for multi-classification task to classify leukemia subtypes using the Acute Lymphoblastic Leukemia (ALL) dataset, which comprises a diverse collection of 3,256 multi-cell blood smear images classed according to their maturity subtypes. In this project, we also proposed a new Convolutional Neural Network (CNN) + U-Net architecture [12], and evaluated with multiple machine learning models, including SVM, XGBoost, VGG16, DenseNet- 201, and ConvNeXt. We focused on optimizing accuracy while considering the misclassification costs associated with different leukemia stages, and found that ConvNeXt and DenseNet-201 pretrained model combined with our proposed architecture performs the best, with DenseNet-201 topping in performance across accuracy and having minimal misclassification costs.

##  Dataset

The models in this study will be using the archived dataset of Acute Lymphoblastic Leukemia (ALL) blood cancer, which provides a large collection of 3,242 peripheral blood smear (PBS) images, organized into two subfolders where one contains the original images captured, and the other one pre-segmented. Within each of these two subfolders contains folders which hold images based on developmental subtypes: Benign, Early Pre-B, Pre-B, and Pro-B ALL. The dataset was prepared in the bone marrow laboratory of Taleqani Hospital in Tehran, Iran [6].

<img src="images/pre_images.png" alt="Sample Prediction" width="500"/>
<img src="images/pro_images.png" alt="Sample Prediction" width="500"/>

The images displayed clear differentiation between classes especially between Early and Pre, as well as Pro which shows a sudden lightness in colour tone of the cells, reflecting variations in cellular morphology that could be critical for accurate classification. This visual assessment is crucial for confirming the relevance of the dataset to the research objectives and the potential for successful application in automated diagnostic systems. From the exploratory data analysis of the image data and resolution, the only aspect we need to focus on is rebalancing the datas scarcity and unequal sizes in each class via augmentation techniques.

---

## Features

- Multi-cell spatial feature analysis
- CNN-based custom architecture
- Transfer learning with pretrained models (VGG16, DenseNet-201, ConvNeXt)
- Evaluation using classification accuracy and misclassification cost metrics

---

## Proposed CNN Model Architecture

<img src="images/proposed_architecture_flowchart.png" alt="Sample Prediction" width="500"/>

This architecture combines a pretrained CNN backbone with a U-Net-inspired segmentation block for joint segmentation and classification of leukemia cells. The input image is first converted to the HSV color space, and a binary mask isolates purple-stained lymphoblasts based on defined HSV thresholds. This mask is applied to extract segmented regions of interest.

The segmentation block follows the U-Net design, with an encoder-decoder structure using convolutional layers, max pooling, and skip connections. The pretrained model (ImageNet weights, include_top=False) extracts high-level features, which are pooled, flattened, and concatenated with features from the segmentation block.

A classification head processes the combined features through dense layers (512 → 256 → 128) with batch normalization, dropout (0.7), and L2 regularization. The final output is a softmax layer with four units corresponding to leukemia maturity stages: Benign, Early, Pre, and Pro.

---

## Results

| Model       | Train Accuracy | Test Accuracy | Total Misclassification Cost | Average Cost Per Sample | Cost-Sensitive Accuracy | Weighted F1-Score |
|-------------|----------------|---------------|------------------------------|--------------------------|-------------------------|-------------------|
| SVM         | 0.9971         | 0.9517        | 345                          | 0.6660                   | 0.9879                  | 0.9448            |
| XGBoost     | 0.9947         | 0.9479        | 380                          | 0.7336                   | 0.9867                  | 0.9419            |
| VGG16       | 0.8637         | 0.9531        | 385                          | 0.7432                   | 0.9865                  | 0.9503            |
| DenseNet-201| 0.9698         | 0.9961        | 20                           | 0.0386                   | 0.9993                  | 0.9958            |
| ConvNeXt    | 0.9927         | 0.9941        | 40                           | 0.0772                   | 0.9986                  | 0.9950            |

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

## Acknowledgements

This work was made possible by the ALL image dataset and the support of deep learning libraries such as Keras, TensorFlow and caret

---

## References

[1] Nan Zhang, Jinxian Wu, Qian Wang, Yuxing Liang, Xinqi Li, Guopeng Chen, Linlu Ma, Xiaoyan Liu,
and Fuling Zhou. Global burden of hematologic malignancies and evolution patterns over the past 30
years. Blood Cancer Journal, 13(1):82, 2023.

[2] World Health Organization et al. Early cancer diagnosis saves lives, cuts treatment costs. World Health
Organization, 2017.

[3] Tibor Mezei, Melinda Kolcsar, András Joó, and Simona Gurzu. Image analysis in histopathology and
cytopathology: From early days to current perspectives. Journal of Imaging, 10(10):252, 2024.
[4] Ahmed J Abougarair, M Alshaibi, Amany K Alarbish, Mohammed Ali Qasem, Doaa Abdo, Othman
Qasem, Fursan Thabit, and Ozgu Can. Blood cells cancer detection based on deep learning. Journal of
Advances in Artificial Intelligence, 2(1):108-121, 2024.

[5] Amjad Rehman, Naveed Abbas, Tanzila Saba, Syed Ijaz ur Rahman, Zahid Mehmood, and Hoshang
Kolivand. Classification of acute lymphoblastic leukemia using deep learning. Microscopy Research and
Technique, 81(11):1310-1317, 2018.

[6] Mustafa Ghaderzadeh, Mehrad Aria, Azamossadat Hosseini, Farkhondeh Asadi, Davood Bashash, and
Hassan Abolghasemi. A fast and efficient cnn model for b-all diagnosis and its subtypes classification
using peripheral blood smear images. International Journal of Intelligent Systems, 37(8):5113-5133,
2022.

[7] Mehmet Erten, Prabal Datta Barua, Sengul Dogan, Turker Tuncer, Ru-San Tan, and UR Acharya.
Concatnext: An automated blood cell classification with a new deep convolutional neural network.
Multimedia Tools and Applications, pages 1-19, 2024.

[8] Murray H Loew. Feature extraction. Handbook of medical imaging, 2:273-342, 2000.

[9] Scott E Umbaugh. Digital image processing and analysis: computer vision and image analysis. CRC
Press, 2023.

[10] David A Forsyth and Jean Ponce. Computer vision: a modern approach. prentice hall professional
technical reference, 2002.

[11] Dengsheng Zhang and Guojun Lu. Review of shape representation and description techniques. Pattern
recognition, 37(1):1-19, 2004.

[12] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
image segmentation. In Medical image computing and computer-assisted intervention-MICCAI 2015:
18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages
234-241. Springer, 2015.

[13] Mingle Xu, Sook Yoon, Alvaro Fuentes, and Dong Sun Park. A comprehensive survey of image aug-
mentation techniques for deep learning. Pattern Recognition, 137:109347, 2023.

[14] Ravindra Singh, Naurang Singh Mangat, Ravindra Singh, and Naurang Singh Mangat. Stratified sam-
pling. Elements of survey sampling, pages 102-144, 1996.

[15] Corinna Cortes. Support-vector networks. Machine Learning, 1995.

[16] GeeksforGeeks. Difference between random forest vs xgboost, 2024. Accessed: 2024-11-18.

[17] Saiwa.ai. Xgboost in machine learning: Everything you need to know, 2024. Accessed: 2024-11-18.

[18] Guestrin Chen. Xgboost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining, pages 785-794, 2016.

[19] Random Realizations. Xgboost explained, 2024. Accessed: 2024-11-18.

[20] The AI Edge. Understanding xgboost: From a to z, 2024. Accessed: 2024-11-19.

[21] Ragav Venkatesan and Baoxin Li. Convolutional neural networks in visual computing: a concise guide.
CRC Press, 2017.

[22] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recog-
nition. arXiv preprint arXiv:1409.1556, 2014.
[23] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected con-
volutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 4700-4708, 2017.

[24] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A
convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 11976-11986, 2022.
