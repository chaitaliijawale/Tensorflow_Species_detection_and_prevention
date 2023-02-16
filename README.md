# Endangered Species Detection, Prevention measures and Global level of Data visualization
This machine learning project aims to detect and prevent endangered species from going extinct. The project leverages computer vision techniques to identify endangered species from images and videos, and provide insights on their population trends and habitats. The project also explores various measures that can be taken to protect these species from further endangerment. This is Semester project for ML subject at Loyalist College in Toronto for Artificial Intelligence and Data Science Course(4000 Victoria Park Ave, Toronto, Ontario M2H 3S7)
 
## 1.1 Dataset
The project uses a dataset of images of various endangered species, including African elephant, Lion, Arctic fox, Cheetah, Panda, Chimpanzee, Panther and Rhino. The dataset of these animals are edited using different image editing softwares such as Adobe, Picasa etc. to balance the dataset from different angles and properties. The dataset is collected from various sources, including wildlife conservation organizations and online repositories.

![Endangered species](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/Top%208%20Extinctive%20animals.png)

## 1.2 Data Preprocessing
Before the dataset can be used for training a machine learning model, it undergoes several preprocessing steps, including image resizing, data augmentation/labelling, and data normalization. The images are resized to a standard size to ensure consistency across the dataset, while data augmentation is used to increase the size of the dataset and prevent overfitting. Data normalization is applied to ensure that the pixel values in the images are in a similar range, which can improve the accuracy of the model. Currently PascalVOC format is used for the image labelling using LabelImg toolkit. 

![Data Annotation](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/data_annotation.gif)

## 1.3 Machine Learning Model
The machine learning model used in this project is a Faster Region based convolutional neural network (Faster-RCNN) that is trained on the preprocessed dataset. The CNN is trained to identify the species in the images, as well as provide insights on their population trends and habitats. The CNN is optimized using various techniques, including hyperparameter tuning and regularization, to improve its accuracy and prevent overfitting. We have used NVIDIA's CUDA toolkit to train the model on local system for high Performance. 

![Faster RCNN](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/Faster_RCNN.png)

### 1.3.1 Why Faster RCNN?
Object Detection is a challenge in Computer Vision. Driven by the success of RCNN & Fast RCNN but still relatively slow.
                                    
| Model | Test time / image | Speed | Mean Average Precision (mAP) |
| -------- | -------- | -------- | -------- |
| R-CNN Model | 50 Sec | 1x | 66.0% |
| Fast R-CNN Model | 2 Sec | 25x | 66.9% |
| Faster R-CNN Model | 0.2 Sec | 250x | 66.9% |

![Faster RCNN Architecture](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/Faster%20RCNN%20Working.png)

## 1.4 Prevention Measures
In addition to detecting endangered species, the project also explores various measures that can be taken to prevent these species from further endangerment. These measures include habitat conservation, wildlife conservation education, and policy advocacy. The project provides insights on how these measures can be implemented and their potential impact on the protection of endangered species.

## 1.5 Global Level Data Visualization
The project also provides global level data visualization of endangered species population trends, habitat distribution, and conservation efforts. The data is presented in various formats, including maps, graphs, and charts, to provide insights on the current state of endangered species and their habitats. The data visualization is created using google looker studio.

## 1.6 Installation
To run the project, you will need to install the following dependencies:(requirment.txt is recomended for Python 3.7.* version)
```
pip install - r requirements.txt
```

Python 3.x
TensorFlow
Keras
NumPy
Pandas
Flask
Once the dependencies are installed, you can clone the repository and run the project using the Jupyter Notebook or your preferred IDE.

Navigate to the source directory for flask server
```
cd root_dir:\tensorflow\models\research\object_detection\pandas-main\pandas-main\src
```
Set the flask environment
```
set flask_env=development
```
Run the flask
```
flask run
```
## 1.7 How this project works?

### 1.7.1 Upload an image to the server to identify the species is endangered or not?
Flask server UI
![UI on flask server](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/select%20and%20upload%20animal%20image.jpg)

### 1.7.2 Upload an  Animal Image
![Upload an  Animal Image](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/2.%20upload%20image.jpg)

### 1.7.3 ML model will identify the species
![species identification using Tensorflow](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/3.%20animal%20detection%20.jpg)

### 1.7.4 Data Visualization report 1
![data visualization for detected species](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/4.%20elephant%20report.jpg)

### 1.7.4 Data Visualization report 2
![data visualization for depletion of species](https://github.com/chaitalijawale08/Tensorflow_Species_detection_and_prevention/blob/950c36256dfb3901f6e6e826f86f65e6a9426ff3/Sample%20Test%20Images/5.%20population%20depletion%20report.jpg)

## 1.8 Conclusion
The Endangered Species Detection and Prevention Measures project is an important step towards protecting endangered species from extinction. By using machine learning techniques, the project can identify endangered species from images and videos, provide insights on their population trends and habitats, and explore various measures that can be taken to prevent further endangerment. The global level data visualization can also provide insights on the current state of endangered species and their habitats, which can inform conservation efforts and policy advocacy.
