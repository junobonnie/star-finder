import cv2
import matplotlib.pyplot as plt
from FloodFill import floodfill, color_maps

def draw_img(img, cmap = 'viridis'):
    plt.figure(dpi=300)
    plt.imshow(img, cmap)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

img = cv2.imread(r"Downloads\Starsinthesky.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
dst = dst//255
draw_img(img)
draw_img(dst, "gray")
clusters = floodfill(dst)
color_maps(dst, clusters)
draw_img(dst, "tab20b")
print(len(clusters))
