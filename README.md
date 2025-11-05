**Project Workflow: Prescription Recognition System**

This project encompasses both the development (training and evaluation) of a deep learning model for handwritten prescription recognition and its deployment as a web application.

**Phase 1: Model Development (Training and Evaluation)**
This phase is detailed in `final_doctors_handwritten_prescription_bd.py` (originally a Jupyter notebook).

1.  **Data Collection & Preparation:**
    *   **Input:** Raw prescription image dataset (e.g., from `/kaggle/input/doctors-handwritten-prescription-bd-dataset/`).
    *   **Labels:** Corresponding CSV files (`training_labels.csv`, `validation_labels.csv`, `testing_labels.csv`) containing `IMAGE` filenames and `MEDICINE_NAME`.
    *   **Preprocessing (`process_folder` function):**
        *   Images are read, converted to grayscale.
        *   **Enhancement:** Adaptive histogram equalization (CLAHE), median blur for noise reduction, and sharpening are applied.
        *   **Resizing & Padding:** Images are resized to a `TARGET_SIZE` (e.g., 320x320) while maintaining aspect ratio, then padded with a median background color to fit the target dimensions.
        *   **Output:** Processed images are saved to a new directory structure (e.g., `/kaggle/working/processed_dataset/`).

2.  **Dataset & DataLoader Setup:**
    *   **`RXWordsDataset` Class:** A custom PyTorch `Dataset` is defined to:
        *   Load preprocessed images and their corresponding medicine names from CSV files.
        *   Use `LabelEncoder` to convert medicine names into numerical labels.
        *   Filter out entries for which image files are missing.
    *   **Image Transformations:**
        *   `train_transform`: Includes augmentations like resizing, color jitter, random rotation, affine transformations, perspective, and Gaussian blur, followed by conversion to tensor and normalization (using ImageNet mean and standard deviation).
        *   `valtest_transform`: Minimal transformations (resizing, tensor conversion, normalization) for validation and testing sets.
    *   **DataLoaders:** `DataLoader` instances are created for training, validation, and testing, enabling batch processing and shuffling.

3.  **Model Architecture & Configuration:**
    *   **Base Model:** A `ResNet18` model is loaded, pre-trained on ImageNet weights.
    *   **Custom Head:** The final fully connected layer (`model.fc`) of ResNet18 is replaced with a new `Linear` layer (and a `Dropout` layer) to output `num_classes` (the number of unique medicine names).
    *   **Device:** The model is moved to an available device (CUDA/GPU if available, otherwise CPU).
    *   **Loss Function:** `CrossEntropyLoss` is used for multi-class classification.
    *   **Optimizer:** `AdamW` is chosen for optimization.
    *   **Scheduler:** `ReduceLROnPlateau` is used to adjust the learning rate during training based on validation accuracy.

4.  **Training & Evaluation Loop:**
    *   **Training:** The model is trained for a specified number of `EPOCHS`. Each epoch involves:
        *   Iterating through the `train_loader`, performing forward and backward passes, and updating model weights.
        *   Calculating training loss and accuracy.
    *   **Validation:** After each training epoch, the model is evaluated on the `val_loader` to monitor performance on unseen data.
    *   **Early Stopping:** Training stops if validation accuracy does not improve for a set number of `PATIENCE` epochs to prevent overfitting.
    *   **Checkpointing:** The model with the best validation accuracy is saved as `best_resnet18.pt`.
    *   **Final Evaluation:** The best-performing model is loaded and evaluated on the `test_loader` to assess its generalization performance.
    *   **Metrics:** `classification_report` and `confusion_matrix` are generated to provide detailed performance analysis.

5.  **Mapping Generation & Export:**
    *   **Label Map:** `label_map.json` is created, mapping numerical class indices back to medicine names.
    *   **Medicine to Generic Mapping:** All label CSVs are combined to create `med2gen_map.json`, which maps each medicine name to its most frequent generic name.
    *   **Prediction Export:** Test predictions, including true and predicted medicine and generic names, are saved to `test_predictions_with_generics.csv`.
    *   **Model Files Export:** The trained model (`best_resnet18.pt`), `label_map.json`, and `med2gen_map.json` are copied to a designated `model_files` directory (which corresponds to the `model/` directory in the Flask application).

**Phase 2: Application Deployment (Web Interface for Inference)**
This phase is handled by `app.py` and `utils/inference.py`.

1.  **Web Application (`app.py`):**
    *   A Flask application is initialized.
    *   It defines an `UPLOAD_FOLDER` (`uploads/`) for storing user-uploaded images.
    *   The root route (`/`) handles both GET (display form) and POST (process upload) requests.
    *   **Image Upload:** Users upload prescription images via an HTML form. The application validates file types and securely saves the image.
    *   **Inference Call:** Upon successful upload, `app.py` calls the `predict` function from `utils/inference.py` with the path to the uploaded image.
    *   **Result Display:** The prediction results (medicine and generic names) are passed to the `index.html` template for display.

2.  **Inference Module (`utils/inference.py`):**
    *   **Model Loading:** Loads the pre-trained `ResNet18` model (`best_resnet18.pt`) and the `label_map.json` and `med2gen_map.json` files from the `model/` directory.
    *   **Preprocessing:** Defines the same image transformations (resize, tensor conversion, normalization) used during model training.
    *   **`predict(image_path)` Function:**
        *   Takes the path to an input image.
        *   Opens and converts the image to RGB.
        *   Applies the necessary transformations.
        *   Performs a forward pass through the loaded PyTorch model to get predictions.
        *   Uses `label_map.json` to convert the predicted index to a medicine name.
        *   Uses `med2gen_map.json` to find the generic name for the predicted medicine.
        *   Returns a dictionary containing the predicted medicine and generic names.

3.  **Frontend (`templates/index.html` & `static/css/style.css`):**
    *   `index.html`: Provides a simple web interface with an image upload form. It dynamically displays the prediction results (medicine and generic names) using Jinja2 templating.
    *   `style.css`: Provides basic styling for the web application.

**End-to-End Project Workflow Diagram:**

<img width="6578" height="1685" alt="image" src="https://github.com/user-attachments/assets/7ec5557e-6644-430c-9c3e-74a064ec2720" />


**Explanation of End-to-End Workflow:**

1.  **Model Development:** The `final_doctors_handwritten_prescription_bd.py` script handles the entire lifecycle of creating the deep learning model. It starts with raw data, preprocesses images, sets up datasets, defines and trains a ResNet18 model, evaluates its performance, and finally generates the necessary `label_map.json`, `med2gen_map.json`, and `best_resnet18.pt` files. These crucial model artifacts are then exported to the `model/` directory.
2.  **Application Deployment:** The Flask application (`app.py`) serves as the user-facing interface. When a user uploads a prescription image via the web browser, `app.py` saves it and then passes it to the `utils/inference.py` module. This module loads the pre-trained model and mappings (created in the Model Development phase) to perform the actual recognition. The predicted medicine and generic names are then sent back to `app.py`, which renders them on the `index.html` page for the user to see.

This two-phase approach clearly separates the machine learning pipeline from the web application, allowing for modular development and easier maintenance.

_completion>
