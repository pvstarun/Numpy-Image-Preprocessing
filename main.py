from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

# Create a checkerboard of resolution 200 with tile size 20 (10x10 tiles)
checker = Checker(resolution=64, tile_size=4)
checker.draw()
checker.show()

# Create a circle of resolution 200 with radus of 50
circle = Circle(resolution=200, radius=25, position=(100, 50))
circle.draw()
circle.show()

# Create a spectrum of 256 resolution
spectrum = Spectrum(resolution=1024)
spectrum.draw()
spectrum.show()

gen = ImageGenerator(
    file_path='/Users/Tarun/Desktop/G Drive/4th Semester/Deep Learning/Exercises/exercise0_material/src_to_implement/data/exercise_data',
    label_path ='/Users/Tarun/Desktop/G Drive/4th Semester/Deep Learning/Exercises/exercise0_material/src_to_implement/data/Labels.json',
    batch_size=250,
    image_size=[256, 256, 3],
    rotation=True,
    mirroring=True,
    shuffle=True
)

# Show a batch
gen.show()
print("Epoch:", gen.current_epoch())
