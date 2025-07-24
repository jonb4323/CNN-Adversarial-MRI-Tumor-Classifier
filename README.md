# CNN-Adversarial-MRI-Tumor-Classifier
This project done entirely in **Jupyter Notebook** builds a convolutional neural network (CNN) to classify brain MRI scans for tumor detection, and evaluates the model's robustness using adversarial attacks.

***Overview***</br>

**Model**: A simple CNN with batch normalization and average pooling layers, trained to classify grayscale MRI scans as tumor/no-tumor.</br>
**Loss/Optimization**: Cross-entropy loss with Adam optimizer.</br>
**Adversarial Testing**: Adversarial examples are generated using a custom attack module and fed to the trained model to observe performance degradation.</br>
**Visualization**: Both original and perturbed images are visualized with model predictions and confidence scores.</br>


***Features***
- Brain MRI binary classification
- Custom training loop using PyTorch
- Adversarial robustness testing
- Visualization of adversarial impact on model predictions

***Files***
**imports.py**: Holds all the imports used in the program (making it easy and simple to change or modify imports)
**image_loader.py**: Loads the regular (png, jpeg, jpg) image, using tansforms to compose each image to -> greyscale, resized to 64x64, and tensor transformation (with their respective label yes/no tumor)
**model_training.py**: This creats the cnn using 2 main layers (Conv, Norm, Pooling) then flattened the image linearly to the possible 2 outcomes 
**attacks.py**: Generated the attack for the images using **advertorch.attacks** with a max distortion of 0.01 eps 
**adversarial_attack.py**: This class calls the attack on a new set of data to prevent overfitting the data and using **matplotlib** to plot the images with their results ater the attack 

**Tested using PyTest**
Projects testing was done using pytest:
  
**Tests Done**:
- Imports
- Dependencies 
- Valid FilePath 
- Valid Images
- Valid CNN Creation (w/ parameter verification)
- Forward Pass
- Training Loop
- Generate Attack
- Adversarial Attack

**Libraries Used**
- Pytorch -> nn, nn.functional (Network building)
- torchvision -> dataloader
- torch.optim -> optimization
- matplotlib -> plt
- PIL (Python Image Library) -> Image
- pytest -> all the testing

 **How to RUN**
 1. Clone the repo and open the Jupyter Notebook or scripts
 2. Adjust image file paths to match your local directories
 3. Train the CNN using model_training.py
 4. Run adversarial_attack.py to test the model with perturbed images
 5. (Optional) Modify the eps value to increase attack strength and observe the model's robustness
    
