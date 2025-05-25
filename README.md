# Brain Tunor Detect 

Deep learning is a rapidly evolving field with significant implications for medical imaging. Currently, the interpretation of medical images, such as MRI and CT scans, is predominantly performed by radiologists and specialized physicians. However, this interpretation process can sometimes be subjective. Even experienced professionals may face challenges in consistently evaluating scans, for instance, when determining the presence or classifying a tumor solely from visual information. Furthermore, medical experts often need to review large volumes of images, which can lead to fatigue and an increased risk of oversight or errors. Consequently, the need for automation and decision support in this domain is increasingly critical.

Traditional machine learning algorithms, such as Support Vector Machines (SVMs), have been employed for tumor detection and classification. However, their effectiveness is often constrained by the assumptions made during manual feature definition, which can result in suboptimal sensitivity and specificity. In contrast, deep learning emerges as an ideal solution because these algorithms can automatically learn complex and hierarchical features directly from raw image data, potentially improving accuracy and objectivity.

One of the major challenges in implementing deep learning algorithms in healthcare is the scarcity of high-quality, labeled medical image data, partly due to patient confidentiality concerns and the intensive labor required for accurate annotation.

This project focuses on developing a Convolutional Neural Network (CNN) based on the Xception architecture. This network is trained to detect and classify common types of brain tumors (glioma, meningioma, pituitary tumor) along with non-tumor cases from MRI scans. The data used for this project is sourced from the "Brain Tumor MRI Dataset" available on Kaggle.

Beyond image classification, this project integrates an AI chatbot, powered by the Gemini API. This chatbot aims to provide initial, general explanations of the classification results, answer users' general questions, and, most importantly, always guide users to consult with specialist doctors for definitive medical diagnoses and advice. The goal is to create a responsible informational support tool that can assist in managing the workload of medical professionals and offer patients an initial, understandable point of reference for their results.


## Dataset
download dataset at : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## How to Use

### 1. Training the Model (Optional)

* If you want to retrain the model yourself, you can use the `Trainingmodel.ipynb` notebook.
* Open this notebook using Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
* Execute the cells in the notebook to preprocess data, build the model, train it, and save the trained model.
* Ensure you have access to the necessary dataset.

### 2. Running the Gradio Web Application

After completing the setup and having the model file:
1.  Open a terminal or command prompt in the root directory of the project (where `app.py` is located).
2.  Activate the virtual environment (if you created one).
3.  Run the application:
    ```bash
    python app.py
    ```
4.  The Gradio application will launch and provide a local URL (usually `http://127.0.0.1:7860` or similar). Open this URL in your web browser.
5.  Upload a brain MRI image to get the classification result and interact with the chatbot.

## Medical Disclaimer

⚠️ **CRITICALLY IMPORTANT NOTE:**
* This application and the information it provides (including image classification results and chatbot responses) are **FOR REFERENCE AND INITIAL INFORMATIONAL SUPPORT PURPOSES ONLY.**
* It **IS NOT** a professional medical diagnostic tool and **ABSOLUTELY DOES NOT REPLACE** examination, diagnosis, consultation, and treatment from qualified doctors or medical professionals.
* All decisions related to health and medical treatment must be made after direct consultation with a specialist doctor.
* The developer is not responsible for any decisions made based on information from this application.

## File Structure (Brief)
