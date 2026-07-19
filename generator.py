import os
import json
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # Load labels
        with open(label_path, 'r') as f: # opens the file and reads 
            self.labels = json.load(f) #converts into a python object

        # Get list of image filenames 
        self.image_ids = list(self.labels.keys()) #54, 13, 2 etc
        self.index = 0
        self.epoch = 0

        if self.shuffle:
            np.random.shuffle(self.image_ids) #shuffles the list of image ids in place

    def next(self):
        images = []
        labels = []

        for _ in range(self.batch_size):
            # Loop through data (wrap around if needed)
            if self.index >= len(self.image_ids):
                self.index = 0
                self.epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.image_ids)

            image_id = self.image_ids[self.index] #54
            label = self.labels[image_id] #1

            img = np.load(os.path.join(self.file_path, f"{image_id}.npy"))

            # Resize
            img = resize(img, self.image_size,anti_aliasing=True) #to smoothen the image resizing especially when downsizing

            # Data Augmentation
            if self.mirroring and np.random.rand() > 0.5:
                img = np.fliplr(img)#flip left to right

            if self.rotation:
                k = np.random.choice([0, 1, 2, 3])  # Rotate 0, 90, 180, 270
                img = np.rot90(img, k=k) #rotate 90 counterclockwise by k times

            images.append(img)
            labels.append(label)
            self.index += 1

        return np.array(images), np.array(labels)

    def current_epoch(self):
        return self.epoch

    def class_name(self, label):
        classes = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
        return classes.get(label, "Unknown")#if the label is not in the classes dictionary, it will return "Unknown" for eg 10 - Unknown

    def show(self):
        images, labels = self.next()
        n = int(np.ceil(np.sqrt(self.batch_size)))

        plt.figure(figsize=(12, 12))
        for i in range(self.batch_size):
            plt.subplot(n, n, i + 1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
            plt.axis('off')
        plt.tight_layout() #to avoid overlap in images
        plt.show()
