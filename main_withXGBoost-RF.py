from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import time
import joblib
from torchsummary import summary
import matplotlib.pyplot as plt 

import medmnist
from medmnist import INFO, Evaluator

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


data_flag = 'bloodmnist'
# data_flag = 'pathmnist'
# data_flag = 'tissuemnist'
# data_flag = 'breastmnist'
# data_flag = 'octmnist'



download = True

NUM_EPOCHS = 1
BATCH_SIZE = 128
lr = 0.0001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)
validation_dataset = DataClass(split='val', transform=data_transform, download=download)


# pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = data.DataLoader(dataset=validation_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


print(train_dataset)
print("===================")
print(test_dataset)
print("===================")
print(validation_dataset)

print(f'validation dataset len = {len(valid_loader)}')
# visualization

train_dataset.montage(length=1)


# montage

# train_dataset.montage(length=20)    

# define a simple CNN model

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            # nn.Dropout(0.45),
            nn.ReLU(),
            nn.Linear(128, 256),
            #  nn.Dropout(0.45),
            nn.ReLU(),
            nn.Linear(256, num_classes))
        self.flatten = nn.Flatten()
    def forward(self, x, mode = 'train'):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        features = x.view(x.size(0), -1)
        if (mode == 'eval'):
            return features
        else:
            x = self.fc(features)
            return x
    

model = Net(in_channels=n_channels, num_classes=n_classes)
inp = torch.rand(3,28,28).to(device)

# summary(model,inp)

model = model.to(device) 
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

mode = 'train'
# train

def compute_accuracy_loss(model, data_loader):
    model.eval()
    curr_loss, correct_pred, num_examples = 0.,0,  0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)
    
        probas = model(features.float())
        loss = criterion(probas, targets)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        curr_loss += loss
    curr_loss = curr_loss / num_examples
    return correct_pred.float()/num_examples * 100, curr_loss



def compute_accuracy_lossv2(model, data_loader):
    all_targets = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        
        curr_loss, correct_pred, num_examples = 0.,0,  0
        for features, targets in tqdm(data_loader):
                
            features = features.to(device)
            targets = targets.to(device)
            targets = targets.squeeze().long()
            
            # Forward pass
            outputs = model(features)
            # Calculate loss (optional, if you want to track the loss)
            loss = criterion(outputs, targets)
           
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            curr_loss += loss.item()
            num_examples += targets.size(0)
            # Update lists
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
           

        curr_loss = curr_loss / len(data_loader)
        # Calculate accuracy
        accuracy1 = accuracy_score(all_targets, all_predictions)
        # print(f"Test Accuracy Accuracy: {accuracy1 * 100:.2f}%")
        return accuracy1 * 100, curr_loss


train_accuracy  = 0.0
valid_loss = 100000.0
for epoch in range(NUM_EPOCHS):    
    train_loss = 0.0

    model.train()
    
    print(f'epoch : {epoch}')
    for inputs, targets in tqdm(train_loader):
        plt.imshow(inputs[11].permute(1,2,0))
        inputs = inputs.to(device)
        targets = targets.to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        if (mode == 'eval'):
            features = model(inputs,mode)
        else:
            outputs = model(inputs)       
       
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(train_loader)
    val_accuracy, val_loss = compute_accuracy_lossv2(model, valid_loader)
    # val_accuracy2, val_loss2 = compute_accuracy_loss(model, valid_loader)
    
    if (val_loss < valid_loss):
        valid_loss = val_loss
        fname= f"XGboost_best_model_epoch_{NUM_EPOCHS}.pt"
        torch.save(model,fname)
        print(f"best model accuracy = {val_accuracy}")
    # train_loss = train_loss / 100
    print(f'train loss = {train_loss}')
    print(f'validation loss = {val_loss}, validation accuracy = {val_accuracy}')

# model = torch.load('model.pt')
# Evaluate the model on the test set

# Record the start time
start_time = time.time()
all_targets = []
all_predictions = []
model = torch.load(f"XGboost_best_model_epoch_{NUM_EPOCHS}.pt") #best model load
model.to(device)
model.eval()
with torch.no_grad():
    for inputs, targets in tqdm(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss (optional, if you want to track the loss)
        # loss = criterion(outputs, targets)

        # Get predictions
        _, predicted = torch.max(outputs, 1)

        # Update lists
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
# Calculate accuracy
accuracy1 = accuracy_score(all_targets, all_predictions)
print(f"Test Accuracy Accuracy: {accuracy1 * 100:.2f}%")
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed cnn test time: {elapsed_time} seconds")

with open(f"XGboost_and_RF_executionTime_epoch_{NUM_EPOCHS}.txt", "a") as file:
    file.writelines(f"Test Accuracy for CNN: {accuracy1 * 100:.2f}%\n")
    file.writelines(f"CNN Elapsed  test time: {elapsed_time} seconds\n")
    


# Create a confusion matrix
conf_matrix = confusion_matrix(all_targets, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=np.arange(n_classes))
disp.plot()
disp.ax_.set_title(f"Accuracy-With CNN: {accuracy1 * 100:.2f}%")
disp.figure_.savefig(f'confusion_matrix_with_CNN_classifier_epoch_{NUM_EPOCHS}.png')







# get features from cnn fc layer for rf

def feature_extractor(model, dataloader, mode = 'eval'):
    model.eval()
    embedding_features = []
    embedding_labels = []
    count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if (mode == 'eval'):
                features = model(inputs,mode)
            else:
                outputs = model(inputs)
            embedding_features.append(features.cpu().detach().numpy())
            embedding_labels.append(targets.cpu().detach().numpy())
            # if (count > 100):
            #     break
            # count += 1
    
    RF_embedding_features = np.concatenate(embedding_features, axis=0)
    RF_embedding_labels = np.concatenate(embedding_labels, axis=0)
    return RF_embedding_features, RF_embedding_labels


RF_embedding_features_train, RF_embedding_labels_train = feature_extractor(model, train_loader, mode = 'eval')
RF_embedding_features_test, RF_embedding_labels_test = feature_extractor(model, test_loader, mode = 'eval')

RF_embedding_labels_train = RF_embedding_labels_train.squeeze()
RF_embedding_labels_test = RF_embedding_labels_test.squeeze()

# Train a Random Forest classifier using the extracted features
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(RF_embedding_features_train, RF_embedding_labels_train)



# Record the start time
start_time = time.time()




# Make predictions on the test set
predictions = rf_classifier.predict(RF_embedding_features_test)




# Calculate accuracy
accuracy = accuracy_score(RF_embedding_labels_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed RF test time: {elapsed_time} seconds")
# Create a confusion matrix

with open(f"XGboost_and_RF_executionTime_epoch_{NUM_EPOCHS}.txt", "a") as file:
    file.writelines(f"Test Accuracy for RF: {accuracy * 100:.2f}%\n")
    file.writelines(f"RF Elapsed  test time: {elapsed_time} seconds\n")


# Create a confusion matrix
conf_matrix = confusion_matrix(RF_embedding_labels_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=np.arange(n_classes))
disp.plot()
disp.ax_.set_title(f"Accuracy-With RF: {accuracy * 100:.2f}%")
disp.figure_.savefig(f'confusion_matrix_with_RandomForestClassifier_{NUM_EPOCHS}.png')


# save
joblib.dump(rf_classifier, f"random_forest_epoch_{NUM_EPOCHS}.joblib")
# kaydedilen rf modelini yüklemek ve predict te kullanmak için aşağıdaki kod gerekli..
# load
loaded_rf = joblib.load(f"random_forest_epoch_{NUM_EPOCHS}.joblib")
#predictions = loaded_rf.predict(RF_embedding_features_test)

# Train a XGboost  classifier using the extracted features
model_xgboost = xgb.XGBClassifier()
# Train the model on training data
model_xgboost.fit(RF_embedding_features_train, RF_embedding_labels_train) 





# Record the start time
start_time = time.time()




# Make predictions on the test set
predictions = model_xgboost.predict(RF_embedding_features_test)




# Calculate accuracy
accuracy = accuracy_score(RF_embedding_labels_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed XGboost test time: {elapsed_time} seconds")
# Create a confusion matrix

with open(f"XGboost_and_RF_executionTime_epoch_{NUM_EPOCHS}.txt", "a") as file:
    file.writelines(f"Test Accuracy for XGboost: {accuracy * 100:.2f}%\n")
    file.writelines(f"XGboost Elapsed  test time: {elapsed_time} seconds\n")


# Create a confusion matrix
conf_matrix = confusion_matrix(RF_embedding_labels_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=np.arange(n_classes))
disp.plot()
disp.ax_.set_title(f"Accuracy-With XGboost: {accuracy * 100:.2f}%")
disp.figure_.savefig(f'confusion_matrix_with_XGboost_classifier_{NUM_EPOCHS}.png')

#Save model for future use
filename = f"xgboost_model_epoch_{NUM_EPOCHS}.sav"
pickle.dump(model_xgboost, open(filename, 'wb'))

################################################################

#Load model.... 
loaded_model = pickle.load(open(filename, 'rb'))



