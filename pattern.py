import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution: int, tile_size: int):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size.")

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        num_tiles = self.resolution // self.tile_size

        # Create the base 2x2 tile pattern using zeros and ones
        base_tile = np.array([[0, 1], [1, 0]])

        # Repeat the base pattern to fill the grid
        full_pattern = np.tile(base_tile, (num_tiles // 2, num_tiles // 2))

        # Expand each tile to the correct pixel size
        expanded_tiles = np.kron(full_pattern, np.ones((self.tile_size, self.tile_size)))

        self.output = expanded_tiles
        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("Call draw() before show().")
        plt.imshow(self.output, cmap='gray')
        plt.title("Checkerboard Pattern")
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position  # (x, y)
        self.output = None

    def draw(self):
        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        xx, yy = np.meshgrid(x, y)

        cx, cy = self.position
        # Compute binary mask of the circle using Euclidean distance
        mask = (xx - cx)**2 + (yy - cy)**2 <= self.radius**2

        self.output = np.zeros((self.resolution, self.resolution))
        self.output[mask] = 1
        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("Call draw() before show().")
        plt.imshow(self.output, cmap='gray')
        plt.title("Binary Circle Pattern")
        plt.axis('off')
        plt.show()



class Spectrum:
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = None

    def draw(self):
        # Create gradients for x and y axes
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)

        # Use broadcasting to create 2D planes
        red_channel = np.tile(x, (self.resolution, 1))         # horizontal gradient
        green_channel = np.tile(y[:, np.newaxis], (1, self.resolution))  # vertical gradient
        blue_channel = 1.0 - red_channel  # complementary to red for visual contrast

        # Stack into an RGB image
        self.output = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("Call draw() before show().")
        plt.imshow(self.output)
        plt.title("RGB Color Spectrum")
        plt.axis('off')
        plt.show()