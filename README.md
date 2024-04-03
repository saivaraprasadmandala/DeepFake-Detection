
# DeepFake Detection

This project implements a deep learning-based image classification model for detecting real or fake faces, specifically designed for facial image manipulation detection. It integrates **face detection with MTCNN (Multi-task Cascaded Convolutional Networks)**, **face recognition with InceptionResnetV1**,**and explainability visualization with Grad-CAM (Gradient-weighted Class Activation Mapping)**. The model is fine-tuned for **binary classification (real or fake)** and provides confidence scores for each prediction.







## Technologies and Components

**1)Face Detection with MTCNN:** MTCNN identifies faces in images and outputs bounding box coordinates to extract the face region.

**2)Face Recognition with InceptionResnetV1:** InceptionResnetV1, pretrained on the VGGFace2 dataset, is loaded and fine-tuned for binary classification.

**3)Model Checkpoint Loading:** Pre-trained model checkpoint (resnetinceptionv1_epoch_32.pth) is loaded to initialize the InceptionResnetV1 model.

**4)Image Preprocessing:** Input images are preprocessed using MTCNN-detected face bounding boxes and resized to (256, 256) for model input.

**5)Model Inference:** The model takes input images, detects faces using MTCNN, preprocesses faces, and performs inference. It generates predictions and provides confidence scores for each class.

**6)Grad-CAM for Explainability:** PyTorch Grad-CAM generates heatmaps highlighting regions contributing most to model predictions, aiding in explaining classification decisions.

**7)Gradio Interface:** Gradio creates a simple web interface for testing the model. Users upload images and view predictions along with explainability visualizations.

**8)Prediction and Confidence Calculation:** The model predicts whether input faces are real or fake, and confidence scores for each prediction are calculated and returned.

## Process of Execution


**For Windows OS,**

*	Download, extract the **deepfake-detetction** zip file from the above files. Do ensure to extract the zip file before running it.

*	After successful installation of the kit,search for the **windows batch file** in the above zip file and run the **batch file**. 

* press **'Y'** to run the kit and execute cells in the notebook.

*	To run the kit manually, press **N** and locate the zip file **deepfake-detection.zip.**

*	Extract the zip file and navigate to the directory **deepfake-detection**.

*	Open command prompt in the extracted directory **deepfake-detection** and run the command **jupyter notebook**.

*	Locate and open the **Deepfake_detection.ipynb** notebook from the Jupyter Notebook browser window.

*	Execute cells in the notebook.

**For other Operating Systems,**

•	Download and extract the **kit_installer** zip file.

•	Follow the above execution information to begin set-up. 

•	This **1-click kit** has all the required dependencies and resources to build your DeepFake Detection Engine.


Output screens represent the expected results and visual feedback from the deep fake detection system during testing.When we run the application,a Jupyter source file will be opened and screen appears as follows:

![1](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/2a1c4e16-9a92-42cf-9da0-f96ba59e4167)

Now we need to run all the cells to activate the user interface (Gradio Interface)

![2](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/0904402d-05eb-48f9-8e91-675b8d18bac1)


At the bottom of the page the following interface will be activated :

![3](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/e8e56fad-6049-4e9d-a26f-238c0c2dac1e)


Now you have an option to drop or choose a picture from your system for detection and click the submit button for final results : 

i) Lets check for a DeepFake Image

![4](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/cb35ede3-702f-4ceb-adaa-e1a8aaa5bdbe)


As we all know the above picture is of Elon Musk , which is really deepfake image , and our model predicts it very accurately. 
The results are as follows :

![5](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/f3a85e76-bc26-4ef6-b1cf-215a212bfe55)


ii)Now we will be testing a real image:

![6](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/d851f2dc-f299-4dbf-82b5-70b6e8444a17)



## Acknowledgements

 - [MTCNN](https://github.com/ipazc/mtcnn)
 - [InceptionResnetV1](https://github.com/timesler/facenet-pytorch)
 - [Gradio]( https://gradio.app/)
 - [PyTorch]( https://pytorch.org/)
 - [PyTorch Grad-Cam]( https://github.com/jacobgil/pytorch-grad-cam)
 - [VGGFace2]( https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b)


## Applications

This DeepFake Engine is best suitable for:

* **Social Media Platforms:** Identify and flag manipulated images or videos to prevent the spread of misinformation.

* **News Verification:** Verify the authenticity of images and videos used in news articles and reports.

* **Law Enforcement:** Detect DeepFake content used in criminal activities such as identity theft and cyberbullying.

* **Entertainment Industry:** Protect intellectual property by identifying and preventing unauthorized distribution of DeepFake content.

* **Personal Security:** Verify the authenticity of shared images and videos to avoid falling victim to scams or manipulation tactics.

## Authors

- [@saivaraprasadmandala](https://www.github.com/saivaraprasadmandala)


## Feedback

If you have any feedback or questions, please reach me out at mandalasaivaraprasad@gmail.com

