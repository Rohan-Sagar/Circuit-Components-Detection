# Object-Detection-Circuit-Components

## Overview
This repository contains the work done for the final project in ECE 49595 Computer Vision, instructed by Dr. Jeffrey Mark Siskind. The project focuses on developing an object detection system for circuit components, starting with resistors. It employs advanced computer vision techniques and machine learning algorithms for accurate detection.

## Team Members
- Rohan Sagar
- Pranav Chintareddy

## License
This project is licensed for educational use only.

## Project Description

### Objective
The primary objective of this project is to create an efficient and reliable method for detecting resistors in circuit images using polygon segmentation. The project leverages PyTorch, a leading deep learning framework, to train a model capable of identifying and localizing resistors with high accuracy.

### Dataset Preparation
- **Annotation**: Utilizing MakeSense.ai, we annotated resistors using Polygon segmentation. The annotated categories included 'Resistor' and 'Wire'. Annotations were exported in COCO JSON format to ensure compatibility with various deep learning frameworks.
- **Image Split**: We divided our dataset into training and validation sets using a 90-10 split. This ratio ensures a substantial amount of data for training while retaining a significant portion for model validation.

### Model Training
- **Custom DataLoader**: Developed a custom DataLoader in PyTorch to handle our specific dataset. This DataLoader is responsible for processing images, creating masks, and generating bounding boxes around detected objects.
- **Neural Network**: We employed the Faster-RCNN architecture, known for its efficiency and accuracy in object detection tasks.
- **Optimizer**: The model was optimized using the Stochastic Gradient Descent (SGD) optimizer, a popular choice for training deep neural networks due to its effectiveness in converging on optimal solutions.

### Results
- **Training Image 1**: ![Training Image](data/example/Train-sample1.jpeg)
- **Training Image 2**: ![Training Image](data/example/Train-sample2.jpeg)
- **Testing Image**: ![Test Image](data/example/Test-sample.jpeg)

## Installation and Usage

### Prerequisites
Ensure you have Python installed on your system. This project was developed using Python 3.8, but it should be compatible with other Python 3 versions.

### Setting Up the Environment
1. Clone the repository: git clone https://github.com/Rohan-Sagar/Object-Dectection-Circuit-Components.git

### Installing Dependencies
Install the required libraries using pip. It's recommended to use a virtual environment to avoid conflicts with other projects.

1. Create a virtual environment (optional): python -m venv venv source venv/bin/activate
2. Install the required packages: pip install numpy opencv-python pandas torch torchvision Pillow matplotlib

### Running the Notebook
1. Launch Jupyter Notebook: jupyter notebook

2. Navigate to the `code.ipynb` file in the Jupyter Notebook interface and open it.
3. Run the cells in the notebook to execute the code.

### Usage
- The notebook contains code to train and evaluate the object detection model.
- Follow the instructions and comments in the notebook to use different functionalities like training, testing, and visualizing the results.

### Note
- Ensure that all the required data and image files are placed in the correct directories as specified in the notebook.
- Modify paths in the code if your directory structure is different.

## Contributions
We welcome contributions and suggestions to improve this project. Please feel free to raise issues or submit pull requests.

## Acknowledgements
Special thanks to Dr. Jeffrey Mark Siskind for his guidance and support throughout the course of this project.

## Contact
For any inquiries, please contact:
- Rohan Sagar: [rsagar@purdue.edu]
- Pranav Chintareddy: [pchintar@purdue.edu]

