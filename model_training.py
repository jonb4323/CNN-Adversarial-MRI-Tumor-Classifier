differntly from imports import *
from image_loader import load_images

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), #Flattening the 3d feature to 1d 
            nn.Linear(32 * 15 * 15, 64), 
            nn.ReLU(),
            nn.Linear(64, 2) #two class outcomes (y/n) (keep this in as it helps with preformance)
        )
    #forward propagation
    def forward(self, x): #applying the forward pass connecting layers to form class logits (y/n)
        x = self.cnn_model(x)
        x = self.fc(x)
        return x
        
def loss_function():
    return nn.CrossEntropyLoss() #This is to determine what is wrong (like hitting yo dog with a newspaper)
        
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001) #learning rate of 0.001 since we want it to be somewhat accurate

def training_loop(model, all_data, epoch, device='cpu'):
    criterion = loss_function()
    optim = optimizer(model)

    model.to(device)
    model.train()
    
    for epoch in range(epoch):
        running_loss = 0.0
        if epoch == 0:
            print("Loading...")
        for images, labels in all_data:
            images, labels = images.to(device), labels.to(device).long()
                
            optim.zero_grad() #reset gradients 
            out = model(images)
            loss = criterion(out, labels) #find loss
            loss.backward() #backpropagation
            optim.step() #update weights 

            running_loss += loss.item()
            
        if epoch == 1:
            print("Loading...")
            print(f"Loss: {running_loss/len(all_data):.4f}")
        if epoch == 5:
            print("Loading...")
            print(f"Loss: {running_loss/len(all_data):.4f}")
        if epoch == 10:
            print("Loading...")
            print(f"Loss: {running_loss/len(all_data):.4f}")
        if epoch == 25:
            print("Loading...")
            print(f"Loss: {running_loss/len(all_data):.4f}")
        if epoch == 45:
            print("Loading...")
            print(f"Loss: {running_loss/len(all_data):.4f}")
    print(f"Epoch: {epoch+1}, Loss: {running_loss/len(all_data):.4f}")


model = CNN()
images, dataset_classes = load_images(batch_size=32)

#Changed Epoch from 50 to 1 for testing change back to do actual test 
training_loop(model, images, epoch=1) 