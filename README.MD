# Deep Learning Training Program

This repository contains a Python script (`Deep Learning.py`) designed for training deep learning models using TensorFlow and Keras. The script supports both classification and regression tasks, with configurable parameters for model architecture, optimization, and training.

## Features

- **Supports Classification and Regression**: Automatically adjusts the model and loss function based on the dataset type.
- **Customizable Parameters**: Easily configure the number of layers, neurons, activation functions, optimizers, and more.
- **Early Stopping**: Stops training when the validation loss stops improving.
- **Data Preprocessing**: Includes one-hot encoding, scaling, and train-test splitting.
- **Visualization**: Generates detailed graphs for training metrics (accuracy, loss, etc.).
- **Example Datasets**:
  - `knn_purchase_history.csv` for classification.
  - `california_housing.csv` for regression.

---

## Requirements

✨Program automatically installs missing libraries. In order to install them maually, please follow the instructions below:
### Python Libraries

Install the required libraries using pip:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

### TensorFlow Version

- Recommended TensorFlow version: `2.11` or lower (for GPU support).

---

## Usage

### 1. Dataset Preparation

Ensure the required datasets are available in the same directory as the script:
- `knn_purchase_history.csv` (for classification tasks).
- `california_housing.csv` (for regression tasks).

Alternatively, you can modify the script to load datasets from other locations.

### 2. Running the Script

Run the script using Python:

```bash
python "Deep Learning.py"
```

### 3. Selecting the Task

When prompted, select the task:
- `1` for Classification.
- `2` for Regression.

---

## Configuration

You can customize the following parameters in the script:

### Model Parameters
- `NEURON_COUNT`: Number of neurons per layer.
- `HIDDEN_LAYERS`: Number of hidden layers.
- `ACTIVATION`: Activation function (e.g., `relu`, `sigmoid`).
- `OPTIMIZER`: Optimizer (e.g., `adam`, `sgd`).
- `LOSS_ALGORITHM`: Loss function (e.g., `sparse_categorical_crossentropy`, `mean_squared_error`).
- `METRICS`: Metrics to evaluate (e.g., `accuracy`, `mean_absolute_error`).

### Training Parameters
- `EPOCHS`: Number of training epochs.
- `VALIDATION_SPLIT`: Fraction of data used for validation.
- `TEST_SPLIT`: Fraction of data used for testing.
- `LEARNING_RATE`: Learning rate for the optimizer.

### Early Stopping
- `EARLY_STOPPING_PATIENCE`: Number of epochs to wait for improvement.
- `EARLY_STOPPING_MIN_DELTA`: Minimum improvement required to reset patience.

---

## Output

### Training Metrics

The script displays the following metrics during training:
- Loss
- Accuracy (for classification)
- Mean Absolute Error (for regression)

### Graphs

After training, the script generates graphs for:
- Accuracy (or MAE) over epochs.
- Loss over epochs.

### Model Evaluation

The script evaluates the model on the test set and displays:
- Test loss.
- Test accuracy (or MAE).

### Predictions

The script demonstrates predictions using example data:
- For regression, it predicts house prices.
- For classification, it predicts purchase probabilities.

---

## Example Output

### Regression

```plaintext
✨ Predicted House Price: $325,000
```

### Classification

```plaintext
✨ Predicted Purchase Probability: 85.23%
```

---

## Notes

- Ensure the datasets are formatted correctly (e.g., no missing values).
- Modify the `ColumnTransformer` section for custom preprocessing.
- For GPU support, install the appropriate TensorFlow version and drivers.

---

## Author

Prepared by **Dr. Mustafa AFYONLUOĞLU**  
Date: September 2025  
[https://afyonluoglu.org/](https://afyonluoglu.org/)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.