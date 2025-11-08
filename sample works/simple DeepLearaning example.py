from sklearn.datasets import make_circles


# Make 1000 samples 
n_samples = 1000

# Create circles
print("Creating circles...")
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

print(f"\nConvert them into DataFrame...")
# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
print(f"{circles.head(10)}")

print(f"{circles.label.value_counts()}")

print(f"\nVisualizing the circles...")
# Visualize with a plot
import matplotlib.pyplot as plt
# plt.scatter(x=X[:, 0], 
#             y=X[:, 1], 
#             c=y, 
#             cmap=plt.cm.RdYlBu)
# plt.show()

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")
print("Data includes two input features and one output feature.\n")

# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
print(f"Converting data into tensors...")
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
print(f"{X[:5]} \n {y[:5]}")

# Split data into train and test sets
from sklearn.model_selection import train_test_split

print(f"\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(f"Size of train and Test Sets for X: {len(X_train), len(X_test)}, \nSize of train and Test Sets for y: {len(y_train), len(y_test)}")

from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device for training: {device}")

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers capable of handling X and y input and output shapes
        # Input Layer: 2 features (X)
        # 1 Layer with 5 hidden units (neurons),    hyperparameter: 5
        # Output Layer: 1 feature (y)

        # linear --> y = x * W + b  (no sigmaoid or other function applied)
    
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() 

    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

print(f"\nCreating model...")
# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
print(model_0)

# Alternative Model Definiton using nn.Sequential
# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# ).to(device)

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in

print(f"\nCreating loss function and optimizer...")
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

print("Model parameters:\n",
      "Layer 1 weights:\n", model_0.state_dict()["layer_1.weight"], 
      "\nLayer 1 bias:\n", model_0.state_dict()["layer_1.bias"],
      "\nLayer 2 weights:\n", model_0.state_dict()["layer_2.weight"], 
      "\nLayer 2 bias:\n", model_0.state_dict()["layer_2.bias"])

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def make_predictions():
    model_0.eval()  # Modeli değerlendirme (inference) moduna geçirir.
    # Dropout veya Batch Normalization gibi katmanlar, eğitim sırasında farklı davranırken değerlendirme modunda sabit davranır.
    # Modelin ağırlıkları güncellenmez, yalnızca ileri geçiş (forward pass) yapılır.
    
    # View the first 5 outputs of the forward pass on the test data
    print(f"\nCalculating the first 5 outputs of the forward pass on the test data...")
    #  disables gradient calculation, reducing memory consumption and speeding up computations during inference.
    with torch.inference_mode():
        # Get the model outputs (logits) for test data "X_test"
        y_logits = model_0(X_test.to(device))[:5]

    print(f"X_test (first 5 labels):\n {X_test[:5]}")
    print(f"y_test (first 5 labels):\n {y_test[:5]}")
    print(f"Logits (first 5 outputs of the model):\n {y_logits}")

    print("Using sigmoid function (for classification logits) on model logits... (For regression, softmax is used)")
    # Use sigmoid on model logits
    y_pred_probs = torch.sigmoid(y_logits)
    print(f"Prediction probabilities:\n {y_pred_probs}")

    # Find the predicted labels (round the prediction probabilities)
    y_preds = torch.round(y_pred_probs)
    print(f"Predicted labels:\n {y_preds}")

    # In full
    # y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

    # Get rid of extra dimension : remove dimensions of size 1
    print(f"\nGetting rid of extra dimension using squeeze()...")
    sq_y = y_test[:5].squeeze()
    sq_pred = y_preds.squeeze()

    print(f"y_test (squeezed)          : {sq_y}")
    print(f"Predicted labels (squeezed): {sq_pred}")

# Test "UNTRAINED" model
make_predictions()


# Train steps:
# 1. Forward pass
# 2. Calculate the loss
# 3. Optimizer zero grad
# 4. Loss backward (backpropagation)
# 5. Optimizer step (gradient descent)

print(f"\nTraining the model...")
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad() # Modelin parametreleri (ağırlıklar ve biaslar) üzerinde biriken gradyanları sıfırlar.

    # 4. Loss backwards (backpropagation)
    loss.backward()      # Loss fonksiyonunun çıktısına göre, modelin her bir parametresi için gradyanlar hesaplanır.

    # 5. Optimizer step
    optimizer.step()     # Hesaplanan gradyanları kullanarak modelin parametrelerini günceller.

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
# if Path("helper_functions.py").is_file():
#   print("helper_functions.py already exists, skipping download")
# else:
#   print("Downloading helper_functions.py")
#   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#   with open("helper_functions.py", "wb") as f:
#     f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()










