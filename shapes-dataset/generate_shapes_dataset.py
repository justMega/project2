import os
import random
import math
from PIL import Image, ImageDraw

# Constants
IMAGE_SIZE = 224
NUM_IMAGES_PER_SHAPE = 10000
SHAPES = [
    "circle",
    "square",
    "fig_eight",
    "rectangle",
    "triangle",
    "elipse"
]

OUTPUT_DIR = "./2D_Geometric_Shapes_Dataset"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rotate_and_position_shape(image, shape_image, angle, cx, cy):
    """Rotates and positions the shape on the base image."""
    rotated_shape = shape_image.rotate(angle, expand=True)
    shape_width, shape_height = rotated_shape.size
    paste_position = (cx - shape_width // 2, cy - shape_height // 2)
    image.paste(rotated_shape, paste_position, rotated_shape)


def draw_shape(shape, color, size):
    """Draws a specific shape on a new image and returns it."""
    shape_image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE))
    shape_draw = ImageDraw.Draw(shape_image)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    if shape == "circle":
        radius = size // 2
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        shape_draw.ellipse(bbox, fill=color)

    elif shape == "elipse":
        scale1 = random.randint(2,6)
        radius1 = size // scale1
        radius2 = size // (scale1 + random.randint(2,6))
        bbox = [cx - radius1, cy - radius2, cx + radius1, cy + radius2]
        shape_draw.ellipse(bbox, fill=color)

    elif shape == "triangle":
        points = [
            (cx, cy - size // 2),
            (cx - size // 2, cy + size // 2),
            (cx + size // 2, cy + size // 2),
        ]
        shape_draw.polygon(points, fill=color)

    elif shape == "square":
        half_size = size // 2
        bbox = [cx - half_size, cy - half_size, cx + half_size, cy + half_size]
        shape_draw.rectangle(bbox, fill=color)

    elif shape == "rectangle":
        half_width = size // 2
        half_height = int(size * 0.75)
        bbox = [cx - half_width, cy - half_height, cx + half_width, cy + half_height]
        shape_draw.rectangle(bbox, fill=color)

    elif shape == "fig_eight":
        scale1 = random.randint(3,6)
        scale2 = random.randint(3,6)
        radius1 = size // scale1
        radius2 = size // scale2
        bbox1 = [
            cx - radius1,
            cy - radius1 - radius2,
            cx + radius1,
            cy + radius1 - radius2
        ]

        bbox2 = [
            cx - radius2,
            cy - radius2 + radius1,
            cx + radius2,
            cy + radius2 + radius1
        ]
        width = random.randint(3,6)
        shape_draw.ellipse(bbox1, outline=color, width=width)
        shape_draw.ellipse(bbox2, outline=color, width=width)

    return shape_image


def create_image(shape):
    """Creates a single image with a random shape."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "black")
    color = "white"

    size = random.randint(40, 100)

    angle = random.randint(0, 360)

    # Create the shape image and rotate it
    shape_image = draw_shape(shape, color, size)
    max_dim = int(math.sqrt(2) * size)
    cx = random.randint(max_dim // 2, IMAGE_SIZE - max_dim // 2)
    cy = random.randint(max_dim // 2, IMAGE_SIZE - max_dim // 2)

    rotate_and_position_shape(image, shape_image, angle, cx, cy)

    return image


def generate_dataset():
    """Generates a dataset of images for each shape."""
    for shape in SHAPES:
        shape_dir = os.path.join(OUTPUT_DIR, shape)
        os.makedirs(shape_dir, exist_ok=True)
        for i in range(NUM_IMAGES_PER_SHAPE):
            image = create_image(shape)
            image_path = os.path.join(shape_dir, f"{shape}_{i+1}.png")
            image.save(image_path)


if __name__ == "__main__":
    generate_dataset()
    print(f"Dataset generated in {OUTPUT_DIR}")
