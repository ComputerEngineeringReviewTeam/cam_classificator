import os
import sys
from PIL import Image


def split_image(image, size, output_dir):
    """Splits the image into smaller squares, extending edges with the most common color if necessary."""
    width, height = image.size
    cols = (width + size - 1) // size
    rows = (height + size - 1) // size

    for row in range(rows):
        for col in range(cols):
            left = col * size
            upper = row * size
            right = min((col + 1) * size, width)
            lower = min((row + 1) * size, height)

            cropped = image.crop((left, upper, right, lower))
            if cropped.size != (size, size):
                square = Image.new("RGB", (size, size), (255, 255, 255, 255))
                square.paste(cropped, (0, 0))
                cropped = square

            output_path = os.path.join(output_dir, f"{row}_{col}.png")
            (cropped.save(output_path))


def process_images(new_size, src_dir, target_dir):
    """Processes all images in the source directory, splitting and saving to the target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            src_path = os.path.join(src_dir, filename)

            image = Image.open(src_path).convert("RGB")

            image_name, _ = os.path.splitext(filename)
            image_output_dir = os.path.join(target_dir, image_name)
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

            split_image(image, new_size, image_output_dir)


def main():
    if len(sys.argv) != 4:
        new_size = 1000
        src_dir = "temp"
        target_dir = "temp"
        # print("Usage: python script.py <new_image_size> <source_directory> <target_directory>")
        # sys.exit(1)
    else:
        new_size = int(sys.argv[1])
        src_dir = sys.argv[2]
        target_dir = sys.argv[3]

    process_images(new_size, src_dir, target_dir)


if __name__ == "__main__":
    main()
