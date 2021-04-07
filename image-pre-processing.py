from PIL import Image

im = Image.open('test_images/frame4.png')
img = Image.open('test_images/frame4.png').convert('LA')
im.show()
img.show()
print(im.format, im.size, im.mode)
print(img.format, img.size, img.mode)
