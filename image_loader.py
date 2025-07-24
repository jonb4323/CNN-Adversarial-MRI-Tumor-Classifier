from imports import *

fp = r"C:\Users\bjon6\Documents\src\Brain Tumor Data\\"

def load_images(data_path=fp, batch_size=32, image_scale=(64,64)):
    transform = transforms.Compose([ #Converting images to 64x64 greyscale
        transforms.Grayscale(), 
        transforms.Resize(image_scale), 
        transforms.ToTensor()
    ])
        
    dataset = datasets.ImageFolder(data_path, transform=transform)
    load = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return load, dataset.classes #Returning the new greyscale images ready for training

