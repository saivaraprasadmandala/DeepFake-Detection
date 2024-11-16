# 🚀 **DeepFake Detection**

A deep learning-based image classification model designed to detect real or fake faces, specifically for facial image manipulation detection. This system integrates **MTCNN (Multi-task Cascaded Convolutional Networks)** for face detection, **InceptionResnetV1** for face recognition, and **Grad-CAM (Gradient-weighted Class Activation Mapping)** for explainability. The model is fine-tuned for **binary classification (real or fake)** and provides confidence scores for each prediction.

---

## 🌟 **Key Features**

- 🎯 **Face Detection with MTCNN:** Detects faces in images and extracts bounding boxes for face region cropping.
- 🧠 **Face Recognition with InceptionResnetV1:** Fine-tuned on VGGFace2 to classify faces as real or fake.
- 🔍 **Explainability with Grad-CAM:** Generates heatmaps to highlight regions influencing model decisions.
- 🌐 **Gradio Interface:** Offers an interactive, user-friendly interface for testing with real-time predictions.
- 📊 **Confidence Scores:** Provides detailed prediction confidence for better reliability.

---

## 🛠️ **Technologies and Components**

1. **MTCNN:** Identifies faces and outputs bounding box coordinates.
2. **InceptionResnetV1:** A robust face recognition model pretrained on the VGGFace2 dataset.
3. **Pretrained Model Checkpoint:** Utilizes `resnetinceptionv1_epoch_32.pth` for initialization.
4. **Image Preprocessing:** Resizes input images to (256, 256) after face detection.
5. **Model Inference:** Processes input images to generate predictions for real or fake classification.
6. **Grad-CAM:** Enhances explainability with visual heatmaps for prediction areas.
7. **Gradio:** Simplifies user interaction with an intuitive drag-and-drop interface.

---

## 💻 **Installation and Execution**

### **For Windows OS**
1. 📥 Download and extract the **deepfake-detection.zip** file.
2. 🔎 Locate the **Windows Batch File** inside the extracted folder.
3. 🖱️ Run the batch file:
   - Press **Y** to auto-run the kit.
   - Press **N** to set up manually.

4. **Manual Setup Steps:**
   - Extract the zip file and navigate to the **deepfake-detection** directory.
   - Open a command prompt in the directory and run:
     ```bash
     jupyter notebook
     ```
   - Open the **Deepfake_detection.ipynb** notebook from the Jupyter Notebook interface.
   - Execute all cells in the notebook.

### **For Other Operating Systems**
1. 📥 Download and extract the **kit_installer.zip** file.
2. Follow the steps above for manual setup.

> 💡 **Tip:** The **1-click kit** includes all necessary dependencies and resources to run the project effortlessly.

---

## 📐 **How to Use**

1. Launch the application to open the Jupyter Notebook interface.
2. Execute all cells to activate the Gradio-based user interface.
3. Upload or drag-and-drop an image for detection.
4. Click **Submit** to view the results, including predictions and explainability heatmaps.

---

## 🖼️ **Example Outputs**

### **1. DeepFake Detection**
- **Input Image:**  
  ![DeepFake Image](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/cb35ede3-702f-4ceb-adaa-e1a8aaa5bdbe)

- **Prediction Result:**  
  ![Prediction Result](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/f3a85e76-bc26-4ef6-b1cf-215a212bfe55)

### **2. Real Image Detection**
- **Input Image:**  
  ![Real Image](https://github.com/saivaraprasadmandala/DeepFake-Detection/assets/122043773/d851f2dc-f299-4dbf-82b5-70b6e8444a17)

---

## 🌍 **Applications**

This DeepFake Detection Engine is highly suitable for:

- **📱 Social Media Platforms:** Flag manipulated images/videos to curb misinformation.
- **📰 News Verification:** Ensure the authenticity of media in articles and reports.
- **👮 Law Enforcement:** Detect DeepFake content used in identity theft or cybercrimes.
- **🎥 Entertainment Industry:** Protect intellectual property by preventing unauthorized DeepFake distribution.
- **🔒 Personal Security:** Verify shared images/videos to avoid scams or manipulation.

---

## **Acknowledgements**

This project leverages the following open-source tools and frameworks:

- [MTCNN](https://github.com/ipazc/mtcnn)  
- [InceptionResnetV1](https://github.com/timesler/facenet-pytorch)  
- [Gradio](https://gradio.app/)  
- [PyTorch](https://pytorch.org/)  
- [PyTorch Grad-Cam](https://github.com/jacobgil/pytorch-grad-cam)  
- [VGGFace2](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b)

---

## 👨‍💻 **Author**

- [@saivaraprasadmandala](https://github.com/saivaraprasadmandala)

---

## 💌 **Feedback**

If you have any feedback, questions, or suggestions, feel free to reach out via email:  
📧 **mandalasaivaraprasad@gmail.com**
