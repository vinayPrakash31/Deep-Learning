import os.path
import json
import scipy.misc
import numpy as np
import random
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.batch_idx = 0

    def resize_image(self, img):
        if [img.shape[0], img.shape[1]] != self.image_size[0:2]:
            img = np.resize(img,(self.image_size[0], self.image_size[1],3))
        return img

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        self.labels = dict()
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        # Get the values as integer
        self.labels = {int(k): int(v) for k, v in self.labels.items()}
        self.labels = sorted(self.labels.items())

        # ceil value for total batches
        total_batches = -(-(self.labels[-1][0]+1)//self.batch_size)

        batch = []
        labels= []

        for i in range(self.batch_idx,total_batches):
            temp = 0
            for j in range (0,self.batch_size):

                if self.shuffle:

                    rand_num = np.random.randint(0, self.labels[-1][0] + 1)
                    if temp != rand_num:
                        image = np.load(self.file_path + str(rand_num) + '.npy')
                        image = self.resize_image(image)
                        temp = rand_num
                        batch.append(self.augment(image))
                        labels.append(self.labels[rand_num][1])
                    continue
                else:

                    if (i*self.batch_size+j) < (self.labels[-1][0]+1):
                        image = np.load(self.file_path+str(i*self.batch_size+j)+'.npy')
                    else:
                        start_over = (total_batches * self.batch_size) - self.labels[-1][0] - 1
                        for s in range(0, start_over):
                            image = np.load(self.file_path + str(s) + '.npy')
                            image = self.resize_image(image)

                            batch.append(self.augment(image))
                            labels.append(self.labels[s][1])
                        break

                    image = self.resize_image(image)
                    batch.append(self.augment(image))
                    labels.append(self.labels[i*self.batch_size+j][1])
            self.batch_idx += 1
            break
        batch = np.array(batch)
        labels = np.array(labels)
        return batch, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring:
            mirror_rand = np.random.randint(0,2)
            img = np.flip(img,mirror_rand)
        if self.rotation:
            rotate_rand = np.random.randint(1,4)
            img = np.rot90(img,rotate_rand)
        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        b, l = self.next()

        fig, axs = plt.subplots(ncols=3, nrows=4)
        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])
        i = 0
        for x in range(0,4):
            for y in range (0,3):
                axs[x, y].imshow(b[i])
                axs[x, y].set(xlabel=self.class_name(l[i]))
                i += 1
        plt.show()
