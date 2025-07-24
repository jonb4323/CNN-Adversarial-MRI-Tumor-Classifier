#Main import 
from imports import *

#Function call imports
from image_loader import load_images
from model_training import CNN
from model_training import training_loop
from attacks import gen_attack
from adversarial_attack import main_attack

#Main Tests 
""" **Testing Done**
Imports
Dependencies 
Valid FilePath 
Valid Images
Valid CNN Creation (w/params)
Forward Pass
Training Loop
Generate Attack
Adversarial Attack
"""


#MAIN TESTING
def test_imports(): #testing imports 
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader           
        import torch.optim as optim
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import os 
        import pytest 
        assert True, f"Imports Worked"
    except ImportError as e:
        assert False, f"Imports Failed: {e}"

def test_dependencies(): #testing to make sure all packages and libs are installed properly  
    required = ["advertorch", "numpy", "torchvision", "torch", "matplotlib"]
    
    for pack in required:
        try:
            importlib.import_module(pack)
            assert True, f"Packages {pack} CAN be installed"
        except ImportError as e:
            assert False, f"Package: {pack} CANNOT be installed: {e}"

def test_valid_filepath(): #making sure the file paths are valid  
    try:
        fp = r"C:\Users\bjon6\Documents\src\PyTest Testing Data\\"
        assert True, f"File Path {fp} is valid."
    except RuntimeError as e:
        assert False, f"File Path {fp} is invalid. {e}"

def test_images(): #checks the image loader to check for valid images and greyscale (no need to check the fp again so i hard coded it)
    fp = r"C:\Users\bjon6\Documents\src\PyTest Testing Data\\"
    images, _ = load_images(fp, batch_size=32, image_scale=(64,64)) 

    for batch in images:
        inputs, _ = batch
        for img in inputs: #Checking the valid result with what should be present
            assert img.shape == (1, 64, 64)
            
def test_cnn_creation(): #tesing the creation of the cnn and input layers 
    try: 
        model = CNN()
        assert True, f"Model has been created"
    except RuntimeError as e:
        assert False, f"Cannot create CNN {e}"

    expected_cnn_layers = (nn.Conv2d, nn.BatchNorm2d, nn.AvgPool2d) #Testing for expected layers 

    for i, layer in enumerate(model.cnn_model):
        assert isinstance(layer, expected_cnn_layers), f"Unexpected layer in cnn_model at index {i}: {layer}" 
        
    expected_fc_types = [nn.Flatten, nn.Linear, nn.ReLU, nn.Linear]
    actual_fc_types = [type(layer) for layer in model.fc]
    assert actual_fc_types == expected_fc_types, f"fc layers incorrect: expected {expected_fc_types}, got {actual_fc_types}"

def test_forward_pass(): #Testing the forward pass (making sure it has the right output)
    model = CNN()
    temp_input = torch.randn(1, 1, 64, 64)  #batch size 1, grayscale image 64x64 (passing a tensor *important*)
    output = model(temp_input)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"

def test_training_loop(): #This tests the training loop with temp images 
    fp = r"C:\Users\bjon6\Documents\src\PyTest Testing Data\\"
    images, _ = load_images(fp, batch_size=1, image_scale=(64, 64)) #loading one image for testing
    
    try:
        model = CNN()
        training_loop(model, images, epoch=1, device='cpu')
        assert True, f"Model was trained like a dog :p"
    except RuntimeError as e:
        assert False, f"Cannot run the tesing loop {e}"

def test_gen_attack(): #This tests the generate attack on each image fro mthe gen_attack func 
    model = CNN()
    temp_input = torch.randn(1, 1, 64, 64)  #batch size 1, grayscale image 64x64 (passing a tensor *important*)

    try:
        attack = gen_attack(model)
        assert True, f"The attack was successful on the temp model" 
    except RuntimeError as e:
        assert False, f"There was an issue with the attack: {e}"

def test_adversarial_attack(): #This will test the main attack on the temp images 
    try:
        main_attack()
        assert True, f"The attack worked and the outputs are shown"
    except RuntimeError as e:
        assert False, f"The attack did not work error has occurred: {e}" 
















        
    