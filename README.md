# Coin-Detection-using-OpenCV-in-Python
## AIM : 
To detect and visualize the edges and contours of a coin using image processing techniques such as grayscale conversion, blurring, morphological operations, and Canny edge detection in OpenCV.
## PROGRAM:
### NAME : LAAKSHIT D
### REGISTER NUMBER : 212222230071
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv2.imread('./CoinsA.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis("off")

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

plt.figure(figsize=(6, 6))
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.axis("off")

# Adaptive Threshold
thresh = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    10  
)

plt.figure(figsize=(6, 6))
plt.imshow(thresh, cmap='gray')
plt.title('Adaptive Threshold')
plt.axis("off")

# Morphological opening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

plt.figure(figsize=(6, 6))
plt.imshow(opening, cmap='gray')
plt.title('After Morphological Opening')
plt.axis("off")

# Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

plt.figure(figsize=(6, 6))
plt.imshow(dist_transform, cmap='jet')
plt.title('Distance Transform')
plt.axis("off")

# Normalize
dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
dist_norm_uint8 = dist_norm.astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(dist_norm_uint8, cmap='jet')
plt.title('Distance Transform - Normalized')
plt.axis("off")

# Threshold for sure foreground
ret, sure_fg = cv2.threshold(dist_norm_uint8, 180, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 6))
plt.imshow(sure_fg, cmap='gray')
plt.title('Sure Foreground')
plt.axis("off")

# Dilate for sure background
sure_bg = cv2.dilate(opening, kernel, iterations=5)

plt.figure(figsize=(6, 6))
plt.imshow(sure_bg, cmap='gray')
plt.title('Sure Background')
plt.axis("off")

# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

plt.figure(figsize=(6, 6))
plt.imshow(unknown, cmap='gray')
plt.title('Unknown Region')
plt.axis("off")

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

plt.figure(figsize=(6, 6))
plt.imshow(markers, cmap='jet')
plt.title('Markers before Watershed')
plt.axis("off")

markers = markers + 1
markers[unknown == 255] = 0

plt.figure(figsize=(6, 6))
plt.imshow(markers, cmap='jet')
plt.title('Markers adjusted for Watershed')
plt.axis("off")

# Apply watershed
img_watershed = img.copy()
markers_ws = cv2.watershed(img_watershed, markers)

plt.figure(figsize=(6, 6))
plt.imshow(markers_ws, cmap='jet')
plt.title('Watershed Output')
plt.axis("off")

# Draw boundaries
img_boundaries = img.copy()
img_boundaries[markers_ws == -1] = [255, 0, 0]  # Red lines

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_boundaries, cv2.COLOR_BGR2RGB))
plt.title('Boundaries Detected')
plt.axis("off")

# Count coins
unique_markers = np.unique(markers_ws)
unique_markers = unique_markers[(unique_markers != -1) & (unique_markers != 1)]
num_coins = len(unique_markers)

print("Estimated number of coins:", num_coins)

# Display result
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_boundaries, cv2.COLOR_BGR2RGB))
plt.title(f'Final Detected Coins: {num_coins}')
plt.axis('off')

# Simple blob detection approach
img_blur = cv2.GaussianBlur(gray, (11, 11), 0)
ret, thresh2 = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

# Set up SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 500

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(thresh2)

img_keypoints = cv2.drawKeypoints(
    img,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Detected Coins via SimpleBlobDetector")
plt.axis("off")

print(f"Number of coins detected: {len(keypoints)}")

```
## OUTPUT:

<img width="498" height="558" alt="image" src="https://github.com/user-attachments/assets/f82747d6-c187-432d-9a85-4fa80e3e585f" />

<img width="506" height="585" alt="image" src="https://github.com/user-attachments/assets/979b6831-f936-4efc-95b2-5c8647ec1c03" />

<img width="506" height="587" alt="image" src="https://github.com/user-attachments/assets/3a34b9f3-7144-4ae1-8bb1-8cfdf880a308" />


## RESULT :
Thus the program to detect the edges was executed successfully.
