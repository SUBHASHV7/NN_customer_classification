# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1209" height="799" alt="image" src="https://github.com/user-attachments/assets/67507eb2-b971-4e72-ad56-90438d5b2262" />

## DESIGN STEPS

## STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

## STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

## STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

## STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.

## PROGRAM

### Name: SUBHASH V
### Register Number: 212224240163

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=100)



```
```python
#function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs=model(inputs)
          loss=criterion(outputs, labels)
          loss.backward()
          optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

<img width="1191" height="242" alt="image" src="https://github.com/user-attachments/assets/e36c0433-d3f0-494f-91d9-e621db7f076b" />


## OUTPUT



### Confusion Matrix

<img width="649" height="551" alt="image" src="https://github.com/user-attachments/assets/0428c070-5e71-410f-9dca-d5903aabea73" />

### Classification Report

<img width="1211" height="648" alt="image" src="https://github.com/user-attachments/assets/4fb9e763-dca7-4287-9c5f-3530494d4c65" />


### New Sample Data Prediction

<img width="1608" height="173" alt="image" src="https://github.com/user-attachments/assets/6d86b099-4d7b-4bca-ad64-be5342f90011" />

## RESULT
Thus neural network classification model is developded for the given dataset.
