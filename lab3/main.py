import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

url = "https://quiplabs.com/wp-content/uploads/Porcine-circovirus.png"

req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

cv2.imshow("Original Image", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

plt.figure("Intensity Histogram")
plt.hist(gray.ravel(), bins=64, range=(0, 256), edgecolor="black")

plt.title("Image Intensity Histogram")
plt.xlabel("Intensity")
plt.ylabel("Number of Pixels")
plt.grid(axis="y")
plt.show()