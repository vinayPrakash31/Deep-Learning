import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.ndarray(shape=(resolution,resolution),dtype=int)

    def draw(self):
        x = np.zeros(shape=(self.tile_size,),dtype=int)
        y = np.ones(shape=(self.tile_size,), dtype=int)
        x_y = np.concatenate((x,y),axis=0)
        y_x = np.concatenate((y,x),axis=0)
        first_row = np.tile(x_y,(self.tile_size,self.resolution//(2*self.tile_size)))
        second_row = np.tile(y_x, (self.tile_size,self.resolution//(2*self.tile_size)))
        self.output = np.tile(np.concatenate((first_row,second_row)),(self.resolution//(2*self.tile_size),1))
        result = self.output.copy()
        return result

    def show(self):
        plt.figure(num=1)
        plot =plt.imshow(self.output, cmap=plt.get_cmap('gray'))
        plt.show(block=True)

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.ndarray(shape=(resolution,resolution),dtype=bool)

    def draw(self):
        # return the grid of size resolution x resolution
        y, x = np.ogrid[:self.resolution, :self.resolution]
        dist_from_center = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        self.output = dist_from_center <= self.radius
        result = self.output.copy()
        return result

    def show(self):
        plt.figure(num=2)
        plot = plt.imshow(self.output, cmap=plt.get_cmap('gray'))
        plt.show(block=True)
