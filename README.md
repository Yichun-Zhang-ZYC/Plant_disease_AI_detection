# Plant_disease_AI_detection
#### Problem Statement

Plant diseases significantly affect global agriculture, leading to reduced crop yields and posing a threat to food security. Timely and accurate detection of plant diseases is crucial to mitigate these effects. However, traditional methods of disease identification often rely on manual inspection, which is time-consuming, labor-intensive, and prone to human error. With the increasing global population and the need to enhance food production, a scalable and efficient solution for plant disease detection has become essential. The adoption of machine learning models for automating this process offers great potential, but existing models are often limited by insufficient training datasets and their inability to handle variability in image quality and environmental noise.

If this problem remains unsolved, it will hinder the development of sustainable agricultural practices, potentially exacerbating food insecurity in vulnerable regions. Furthermore, inefficient disease detection methods could lead to the excessive use of pesticides, harming the environment and human health. 

#### Motivation

Recent advances in machine learning and computer vision, such as Vision Transformers (ViTs) and convolutional networks like ResNet, have shown remarkable success in image classification. These models can significantly enhance the precision of plant disease detection when paired with robust datasets and innovative augmentation techniques. However, challenges such as dataset diversity and the ability to generalize across unseen conditions persist. Addressing these challenges is vital to develop a system that not only achieves high accuracy but also ensures robustness in real-world applications.

The Segment Anything Model (SAM) provides a novel approach to segmenting and preprocessing images, enhancing data quality and enabling better model training. Integrating SAM with deep learning architectures like ViT and ResNet could pave the way for highly accurate and efficient disease detection systems. This approach aligns with recent research emphasizing the importance of dataset augmentation and advanced vision models for improving agricultural diagnostics【1】【2】.



#### Proposed Approach

We propose a hybrid pipeline integrating Vision Transformers (ViT) and ResNet with the Segment Anything Model (SAM) to address plant disease detection challenges. The choice of ViT and ResNet stems from their complementary strengths: ViT's global attention mechanism excels in capturing high-level features, while ResNet's hierarchical structure effectively captures local patterns. SAM enhances the dataset by segmenting images to focus on disease-relevant regions, improving data diversity and robustness.

Our methodology involves training and fine-tuning both ViT and ResNet on the PlantVillage dataset, consisting of 20,638 images across 15 disease categories. The training process is augmented using techniques such as gradient blur and rotation to improve model generalization. We evaluate models on original and SAM-segmented datasets, exploring their performance under noise and testing their robustness.

Key experimental steps:
1. **Baseline Zero-Shot Testing**: Conducted with pre-trained ViT and ResNet models on the original dataset.
2. **Model Training on Original Dataset**: Fine-tuning the models with standard training techniques to establish benchmark accuracies.
3. **Integration of SAM-Segmented Data**: Incorporating SAM-segmented images into training and testing pipelines to analyze performance improvements.
4. **Comparison of Models**: Analyzing ViT and ResNet performance metrics, including accuracy, robustness under noise, and inference times.



#### Contributions

This study makes the following contributions:
1. **Hybrid Pipeline Development**: We integrate SAM with two advanced vision models (ViT and ResNet) to improve plant disease detection accuracy and robustness.
2. **Performance Comparison of Vision Models**: By comparing ViT and ResNet on both original and segmented datasets, we highlight their respective strengths and limitations, offering insights into their application in agricultural diagnostics.
3. **Enhanced Data Augmentation Techniques**: Incorporating gradient blur and rotation, we achieve better generalization and robustness, particularly in noisy environments.
4. **Robustness and Efficiency**: Our models demonstrate high robustness, with ViT achieving top-1 accuracy of 99.88% and ResNet achieving a comparable performance of 99.40%. The results underscore the effectiveness of SAM in augmenting plant disease datasets.

This research provides a foundation for scalable and efficient plant disease detection systems, contributing to the broader goal of sustainable agricultural practices.



#### References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *International Conference on Machine Learning (ICML)*.
2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *arXiv preprint arXiv:2010.11929*.

