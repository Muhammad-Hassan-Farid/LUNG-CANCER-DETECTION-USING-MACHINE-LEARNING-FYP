
# Lung Cancer Detection using Supervised Machine Learning

Lung cancer is one of the most prevalent and deadly forms of cancer globally, with survival rates heavily dependent on early detection. Conventional diagnostic methods often identify the disease in advanced stages, reducing the effectiveness of treatment. In this project, we propose a machine learning based system for the early detection of lung cancer using computed tomography (CT) scan images. Our approach focuses on leveraging supervised machine learning algorithms, particularly deep learning techniques such as Convolutional Neural Networks (CNNs), to detect lung cancer nodules accurately.  The system was trained on the publicly available LIDC-IDRI dataset, which contains annotated CT scan images. We implemented and evaluated five CNN architectures: VGG-19, ResNet-50, GoogleNet Inception V3, DenseNet-201, and EfficientNet-B2. Each model was fine-tuned to improve its performance on lung nodule detection. Image preprocessing techniques such as resizing, normalization, and grayscale conversion were applied to enhance the quality of input images. The system’s performance was evaluated based on precision, recall, F1-score, sensitivity, and specificity.  The results demonstrate that deep learning models can significantly improve the early detection of lung cancer, providing a non-invasive, efficient diagnostic tool for clinical use. By incorporating such automated systems into healthcare workflows, it may be possible to reduce mortality rates and enhance patient outcomes through timely interventions.

## Aims and Objects
#### The aim of the project is:

•	To develop a machine learning-based lung cancer detection system

#### The primary objectives of the project are:

•	To perform a comprehensive review of machine learning algorithms for lung cancer detection.

•	 To identify an optimal algorithm for lung cancer detection and implement it.

•	 To validate the algorithm performance using the lung image dataset and achieve optimal accuracy.

## Methodology 
### 1. Proposed System
The proposed system will read the DICOM CT-Scan image in Grayscale format, then it will resize the image according to the model, normalize the image, give to the model and give us the prediction.

![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/Capture.PNG?raw=true)
### 2. Data Collection 
Thoracic CT images with annotated lesions for lung cancer screening and diagnosis are part of the commonly used LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative) collection. The development of computer-aided diagnostic (CAD) systems is one area in which this dataset is particularly important for the advancement of research on lung cancer diagnosis. It was created because of cooperation between eight medical imaging firms and seven academic centers, underscoring the value of interdisciplinary approaches to solving difficult healthcare problems. The 1,018 cases in the LIDC-IDRI dataset each have a few thoracic CT scans along with radiologist annotations. Because they recognize and label lesions—abnormal spots in the lungs that can be early indicators of lung cancer—these annotations are essential. Four seasoned radiologists examined each CT scan separately to guarantee the data's accuracy and dependability. They carefully annotated and categorized the lesions according to the possibility that they were cancerous. The dataset is certain to offer a rich amount of data for training and verifying machine learning models, especially those aimed at lung cancer diagnosis, thanks to this multi-reader procedure.

![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/Data%20Sample.png?raw=true)


### 3. Image Preprocessing

#### 3.1 Reading DICOM Images
Medical picture data are typically stored in DICOM (Digital Imaging and Communications in Medicine) formats. Along with the image data, these files also carry crucial metadata, including patient characteristics, image size, modality, and acquisition information. Reading DICOM files is the initial step in processing the CT scans for lung cancer detection. This guarantees accurate interpretation of the raw imaging data and the necessary contextual data before it is placed into your pipeline for additional processing.


#### 3.2 Image Resizing
The next step is to resize the pictures. Depending on the apparatus and parameters used during image acquisition, medical images, like CT scans, can have a wide range of sizes and resolutions. The photos must be scaled to a consistent shape in order to standardize the data for analysis and guarantee that the deep learning models can handle them effectively.

#### 3.3 Image Normalization
In image processing, normalization is an essential preprocessing step, particularly for deep learning applications. Pixel intensity levels in medical imaging, such CT scans, can vary greatly from one image to the next. Normalization is used to place all photos into a common range. In this stage, the pixel intensity values are usually scaled to a specified range (e.g., 0 to 1).


### 4. Selected Models
Selected Models are based on CNN (Convolutional Neural Network), the following are:

#### 4.1 VGG-19
Developed by the University of Oxford's Visual Geometry Group (VGG), VGG-19 is a deep convolutional neural network. It has 19 layers total: 5 max-pooling layers, 3 fully linked layers, and 16 convolutional layers with 3x3 filters. The network is easy to create because it has consistent architecture and employs modest filters to collect minute features. While preserving crucial information, the max-pooling layers aid in reducing the spatial dimensions of feature maps.
Because of its depth and ability to extract precise features, VGG-19—which is well-known for its excellent accuracy on picture classification tasks—performs well on benchmarks like ImageNet. However, it requires a lot of memory and processing power due to its deep architecture and many parameters. Using previously trained models for a variety of specialized tasks is a common application of transfer learning.

![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Models%20Architectures/VGG-19.png?raw=true)

#### 4.2 ResNet-50
The issue of disappearing gradients in extremely deep networks is addressed by the deep convolutional neural network architecture ResNet-50, which incorporates residual learning. It has fifty layers total, comprising pooling, batch normalization, and convolutional layers. Utilizing residual blocks—shortcut connections that eschew one or more convolutional layers—is the fundamental innovation of ResNet-50. By assisting the network in learning identity mappings, these connections facilitate deep network training and improve overall performance. 
For classification, the architecture consists of one fully connected layer and forty-nine convolutional layers. ResNet-50 is renowned for managing challenging picture classification jobs with quickness and efficacy. By reducing the degradation issue, residual connections help train deeper networks. This enables ResNet-50 to attain high accuracy on benchmarks such as ImageNet with comparatively lower computing costs than deeper versions of ResNet.


![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Models%20Architectures/ResNet-50.png?raw=true)

#### 4.3 GoogleNet Inception V3
Convolutional neural network architecture Inception V3 is renowned for its ability to extract intricate information from images while making effective use of computing power. It presents the Inception module, which integrates pooling operations and several convolutional filter types (1x1, 3x3, and 5x5) into a single layer. With this method, the network may reduce dimensionality and gather data at various sizes while maintaining key properties. Factorization techniques, including swapping out huge convolutions for smaller ones, are another feature of Inception V3, which helps in lowering the processing expenses and parameter count.
48 convolutional layers make up the design, and the last fully linked layer is used for classification. Because to its modular architecture and streamlined operations, Inception V3 is more efficient than previous models and achieves excellent accuracy on picture classification tasks such as those in the ImageNet dataset. It is a well-liked option for many computer vision applications due to its capacity to strike a balance between computing efficiency and accuracy.


![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Models%20Architectures/Inception.png?raw=true)


#### 4.4 DenseNet-201
With its distinct dense connection topology, DenseNet-201 is a deep convolutional neural network that improves feature propagation while using fewer parameters. Every layer in DenseNet-201 is connected to every other layer in a feed-forward manner, which means that each layer's input is the concatenation of all feature maps from layers that came before it. The network performs better thanks to this dense connection design, which promotes feature reuse and helps to improve gradient flow during training.
There are 201 layers in the architecture, consisting of transition layers and dense blocks. DenseNet-201 performs better than other comparable deep networks in picture classification tasks because of its effectiveness in extracting and reusing features, which leads to high accuracy with comparatively fewer parameters. It's an excellent option for a variety of computer vision applications due to its effective feature reuse and great performance on benchmarks like as ImageNet.


![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Models%20Architectures/DenseNet-201.png?raw=true
)


#### 4.5 EfficientNet B-2
EfficientNet-B2 is a variant of the EfficientNet family, designed to optimize the balance between accuracy and computational efficiency. It uses a compound scaling method to uniformly scale depth, width, and resolution of the network, which helps achieve better performance without significantly increasing computational costs. EfficientNet-B2 incorporates a series of Mobile Inverted Bottleneck Convolutions (MBConv) and Swish activation functions, which enhance the network's efficiency and accuracy.       
 Compared to previous deep networks, EfficientNet-B2's architecture has a simplified design with fewer parameters and less computing power needed. It maintains a smaller model size and reduces computing cost while achieving good accuracy on picture classification tasks. Because of its effective resource management and ability to balance efficiency and accuracy, EfficientNet-B2 can be applied to a variety of computer vision tasks.


![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Models%20Architectures/EfficientNet.png?raw=true)

### 5. Results
The 5  models training and testing accuracy, recall score, precision score, f1 score, sensitivity and specificity are below:

![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/Results.PNG?raw=true)

The evaluation of the five models in Table 2 and figure 10 shows strong overall performance, with some subtle distinctions. VGG-19 achieved high training (99.94%) and testing accuracy (99.93%), along with excellent precision, recall, and F1 scores, but it slightly underperformed in specificity (98.61%) and cross-validation (98.89%), indicating a minor struggle with false positives. ResNet-50 balanced its metrics well, with slightly lower testing accuracy (99.60%) than VGG-19, but stronger specificity (99.90%) and a decent cross-validation score (98.37%), suggesting better generalization. InceptionV3 had high precision and recall (99.63%) with a marginally better testing accuracy (99.63%) than ResNet-50, but its cross-validation score (95.86%) was the lowest, implying potential overfitting. DenseNet-201 showed impressive performance across most metrics with perfect testing accuracy (99.60%) and excellent specificity (99.90%), though its cross-validation score (98.16%) hinted at slightly lower stability on unseen data. EfficientNet-B2 outperformed the other models in terms of cross-validation (99.57%) and specificity (99.95%) while keeping high accuracy (99.70%) across training and testing, showing excellent generalization and stability, making it the most reliable model in this comparison.
Based on the analysis of the models, VGG-19 shows strong overall performance but struggles with generalization due to lower cross-validation and specificity, making it less dependable in minimizing false positives. ResNet-50 offers a balanced approach with good testing accuracy, specificity, and generalization, making it a robust choice for lung cancer detection. InceptionV3 achieves near-perfect results in most metrics but shows signs of overfitting, as indicated by its low cross-validation score, meaning it may perform inconsistently on new datasets. DenseNet-201 stands out with exceptional testing accuracy and stability across metrics, although its cross-validation performance suggests it may not generalize as well as EfficientNet-B2. Finally, EfficientNet-B2 emerges as the best model overall, with the highest cross-validation score and nearly perfect performance in all other metrics, indicating that it is the most accurate and generalizable model for early lung cancer detection. , while all models perform well, EfficientNet-B2 and DenseNet-201 offer the most consistent and reliable performance for real-world application, with EfficientNet-B2 providing the best balance between accuracy, specificity, and generalization.

#### 5.1 Models Performance During Training
The model’s performance during training provides important insights into how each model learns over time shows in below figures. VGG-19 shows a steady improvement in accuracy during training but exhibits signs of slight overfitting, with its validation accuracy being lower than its training accuracy, indicating that the model performs better on training data than on unseen data. ResNet-50 demonstrates consistent training and validation performance, suggesting better generalization capabilities, but it takes more epochs to converge to high accuracy compared to the other models. InceptionV3, while achieving near-perfect training accuracy, shows a greater gap between training and validation performance, indicating that the model overfits the training data and may struggle with new data. DenseNet-201 performs well throughout training with minimal difference between training and validation accuracy, indicating excellent learning and generalization abilities, although it took longer to converge fully. EfficientNet-B2 stands out as the best-performing model during training, with nearly perfect training and validation curves that closely match, indicating minimal overfitting and strong generalization. Its efficient architecture allows it to learn quickly while maintaining high accuracy across both training and validation sets.
In summary, EfficientNet-B2 shows the most consistent and efficient performance during training, with minimal overfitting and fast convergence, followed by DenseNet-201. InceptionV3 and VGG-19 show some signs of overfitting, while ResNet-50 demonstrates good generalization but requires more training epochs to achieve peak performance.

#### VGG 19 Performance
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/VGG-19%20Performance.PNG?raw=true)
#### ResNet 50 Performance
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/ResNet-50%20Performance.PNG?raw=true)
#### Inception V3 Performance
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/Inception%20V3%20Performance.PNG?raw=true)
#### DenseNet 201 Performance
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/DenseNet-201%20performance.PNG?raw=true)
#### EfficientNet B2 Performance
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/EfficientNet-B2%20Performance.PNG?raw=true)
⋅ All Models Performance comparison
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/ALL%20Models%20Accuracy%20Performace.PNG?raw=true)
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/All%20Models%20Loss%20Performance.PNG?raw=true)

#### 5.2 Confusion Matrix

The confusion matrixs for the five models provides insights into their classification performance for malignant and benign lung nodules. VGG-19 shows strong classification abilities, but its confusion matrix shows a slightly higher rate of false positives (classifying benign nodules as malignant), reflected in its lower specificity. ResNet-50 offers a better balance, with fewer false positives and false negatives, making it more reliable in correctly distinguishing malignant from benign cases. InceptionV3, despite achieving high overall accuracy, shows signs of overfitting as indicated by its nearly flawless confusion matrix on the test set, yet lower cross-validation performance, suggesting that it may not handle unseen data as effectively. DenseNet-201 shows near-perfect classification performance with minimal misclassifications in both malignant and benign cases, but its slightly lower cross-validation score hints at reduced stability on different datasets. EfficientNet-B2 presents the most balanced and consistent confusion matrix, with very few false positives or false negatives, indicating superior accuracy in distinguishing between malignant and benign nodules across a variety of scenarios.
All models show high classification performance, but EfficientNet-B2 and DenseNet-201 have the least misclassification, with EfficientNet-B2 offering the most robust generalization and balanced performance in identifying malignant and benign nodules.

#### VGG 19 Confusion Matrix:
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/VGG-19%20Confusion%20Matrix.PNG?raw=true)

#### ResNet 50 Confusion Matrix:
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/ResNet-50%20Confusion%20Matrix.PNG?raw=true)

#### Inception V3 Confusion Matrix:
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/Inception%20V3%20Confusion%20Matrix.PNG?raw=true)

#### DenseNet 201 Confusion Matrix:
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/DenseNet-201%20Confusion%20Matrix.PNG?raw=true)

#### EfficientNet B2 Confusion Matrix:
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/EfficientNet-B2%20Confusion%20Matrix.PNG?raw=true)

### Results

#### VGG 19 Results
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/VGG-19%20output.png?raw=true)
#### ResNet 50 Results 
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/ResNet%2050%20output.png?raw=true)
#### Inception V3 Results
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/Inception%20V3%20output.png?raw=true)
#### DenseNet 201 Results
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/DenseNet%20outputs.png?raw=true)
#### EfficientNet B2 Results
![App Screenshot](https://github.com/Muhammad-Hassan-Farid/LUNG-CANCER-DETECTION-USING-MACHINE-LEARNING-FYP/blob/master/Results/EfficientNet-B2%20outputs.png?raw=true)

# Conclusion
We envision our suggested method to use deep learning to transform early lung cancer detection. To address the shortcomings of conventional diagnostic techniques, we aim to leverage sophisticated convolutional neural network designs. Our models are trained and assess using the extensive set of annotated thoracic CT images known as the LIDC-IDRI dataset. By carefully preprocessing the pictures and optimizing neural networks, we hope to attain remarkable precision in the detection of early-stage lung cancer using CT scans. With the ability to provide early diagnosis and more effective treatment options, this ground-breaking technology holds the potential to greatly improve patient outcomes.

