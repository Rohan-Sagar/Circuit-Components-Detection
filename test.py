import cv2 as cv
import torch
import torchvision.transforms as transforms
import sys
from matplotlib import pyplot as plt

img = cv.imread("img.jpg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
transform = transforms.Compose([transforms.ToTensor()])
img_as_tensor = transform(img_rgb)
print("Img as tensor", img_as_tensor)

if img is None:
    sys.exit("Could not read the image.")

print(img.shape)

#img_draw = cv.selectROI(img)
#cv.imshow("Bounded Image", img_draw)

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
#plt.show()

