# Oyster Quality Classification - Project Summary

This project focuses on building a binary image classification model to determine the quality of oysters (good vs. bad) using computer vision techniques and deep learning.

# Project Directory: DeepOysterBinary

This folder contains all the necessary components for training and evaluating an image classification model to classify oyster quality as either "good" or "bad".

##  Folder Structure & File Descriptions

- Dataset/  
  → Contains the training and test image folders organized into subfolders based on class (`good`, `bad`).  
  This is the main dataset used for training and evaluating the binary classifier.

- datasetOriginalTest/  
  → Contains a separate, clean (unaugmented) test set used for evaluating the model's robustness on cleaner samples.

- DeepOysterBinary.ipynb  
  → The Jupyter Notebook with all code, experiments, evaluation metrics, visualizations (confusion matrix, Grad-CAM), and step-by-step documentation. This is the main working file.

- README.txt  
  → You're reading it! This file summarizes the project goal, directory layout, progress, and future directions.

- LICENSE.txt  
  → Contains licensing or usage terms, if applicable. You can modify this file to reflect open-source or proprietary use depending on the project's future.

##  Objective:
To develop a model that can accurately classify oyster images into "good" or "bad" categories. This can help improve food safety and automate inspection in aquaculture.

## Progress Overview:
- Started with a custom CNN model and analyzed its performance using accuracy, confusion matrix, and visualizations.
- Used Grad-CAM to understand what parts of the oyster the model focused on.
- Investigated misclassifications through brightness analysis and image-level inspection.
- Switched to transfer learning with MobileNetV2 to improve accuracy and generalization.
- Fine-tuned only the final classifier layer to prevent overfitting and retain pretrained features.
- Achieved a final test accuracy of ~77.3% with balanced precision/recall on both classes.

##  Evaluation Tools:
- Confusion matrix visualization using seaborn
- Classification report (precision, recall, F1-score)
- Grad-CAM for model interpretability

##  Next Steps:
- Test the model on the California Oyster dataset from Kaggle to evaluate generalization.
- Work with project experts to gather more diverse and higher-volume oyster data.
- Expand the model from binary to multi-class classification (e.g., 5 oyster quality levels).
- Explore augmentation and domain adaptation techniques for better transfer across datasets.

##  Technologies Used:
- PyTorch & Torchvision
- Pretrained MobileNetV2
- scikit-learn for evaluation
- Matplotlib & Seaborn for plotting

##  Acknowledgements:
- ChatGPT (OpenAI) for model guidance, debugging, and experimental design feedback.

This project is a work-in-progress aimed at building scalable AI-based inspection systems in aquaculture. Stay tuned for updates!
