# Brain Tumor Detection (CNN + Streamlit)

A learning project: a Convolutional Neural Network (CNN) that predicts **Tumor / No Tumor**
from a brain MRI image, with a Streamlit web app to try it.

> **Not a medical tool.** It is trained on only 253 images and makes mistakes.
> Never use it to diagnose a real patient.

## What is in this folder
| File | Purpose |
|---|---|
| `brain_tumor_cnn.ipynb` | Training notebook: data split, CNN, training, evaluation, saving the model |
| `app.py` | Streamlit app (Home page + Prediction page) |
| `brain_tumor_classifier.keras` | The trained model |
| `requirements.txt` | Libraries needed by the app (also used by Streamlit Community Cloud) |
| `requirements-notebook.txt` | Extra libraries for the training notebook |

## Dataset
*Brain MRI Images for Brain Tumor Detection* (Kaggle): 253 MRI images, 155 with a tumor
and 98 without. The dataset itself is not included in this repository.

## Method
- Stratified 70 / 15 / 15 split into train, validation and test sets
- CNN with data augmentation, 4 Conv2D + MaxPooling blocks, Dropout, class weights and EarlyStopping
- Results on the held-out test set are shown in the notebook (step 10). With a dataset this
  small, accuracy varies roughly between 70% and 85% depending on the split.

## App features
- Upload a brain MRI image and get the predicted class, confidence and probability of each class
- Images that are clearly not MRI scans (for example colour photos, selfies, documents) are
  rejected with a message instead of being classified. This check uses simple image statistics
  (colour, background), so it is helpful but not perfect: a grayscale photo with a dark
  background can still pass.
- Low-confidence predictions show a warning

## Live app
(Add your Streamlit Community Cloud link here after deploying.)

## How to run locally
1. App only: `pip install -r requirements.txt`
   To also run the training notebook: `pip install -r requirements-notebook.txt`
2. Download the Kaggle dataset and place the `yes` and `no` folders in `brain_tumor_dataset/`
   (only needed for retraining)
3. (Optional, to retrain) run all cells of `brain_tumor_cnn.ipynb`
4. `streamlit run app.py`

If the app cannot load the model (different TensorFlow version), run the notebook once. It
saves a fresh `brain_tumor_classifier.keras` that matches your installation.
