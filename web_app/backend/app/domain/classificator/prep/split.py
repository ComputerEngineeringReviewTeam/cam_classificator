from PIL import Image

def split_image(image, size):
    """Splits the image into smaller squares, extending edges with white color if necessary."""
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
            yield cropped
