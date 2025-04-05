from PIL import Image, ImageDraw
import os

# Create a directory for static files if it doesn't exist
os.makedirs('static', exist_ok=True)

# Create a 32x32 pixel image with transparent background
img = Image.new('RGBA', (32, 32), color=(0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Define colors
husky_gray = (170, 170, 190)
husky_white = (250, 250, 255)
husky_black = (40, 40, 40) 
husky_blue = (100, 140, 220)

# Draw husky face
# Fill the basic shape (gray body)
for y in range(8, 28):
    for x in range(7, 25):
        draw.point((x, y), husky_gray)

# Draw white face marking
for y in range(12, 26):
    for x in range(10, 22):
        draw.point((x, y), husky_white)

# Draw the ears
for x in range(5, 9):
    for y in range(5, 12):
        draw.point((x, y), husky_gray)
for x in range(23, 27):
    for y in range(5, 12):
        draw.point((x, y), husky_gray)

# Draw eyes
draw.rectangle([10, 15, 13, 18], fill=husky_black)
draw.rectangle([18, 15, 21, 18], fill=husky_black)

# Draw blue highlights to eyes
draw.point((11, 16), husky_blue)
draw.point((19, 16), husky_blue)

# Draw nose
draw.rectangle([14, 20, 17, 22], fill=husky_black)

# Save as ICO file
img.save('static/favicon.ico')

print("Husky favicon created at static/favicon.ico") 