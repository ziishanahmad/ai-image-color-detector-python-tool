# AI Image Color Detector Python Tool

This project is an AI-powered tool designed to detect and extract prominent color codes from any given image. By utilizing K-means clustering, the tool identifies the most significant colors in the image and provides their RGB and HEX codes. This Python-based tool is perfect for developers, designers, and anyone interested in analyzing the color composition of images.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ziishanahmad/ai-image-color-detector-python-tool.git
    cd ai-image-color-detector-python-tool
    ```

2. **Install the necessary libraries**:
    ```bash
    pip install opencv-python-headless matplotlib scikit-learn
    ```

## Usage

To run the script and use the tool, follow these steps:

1. **Run the script**:
    ```bash
    python color_detector.py
    ```

2. **Upload an image**:
    - When prompted, upload the image you want to analyze.

3. **View the results**:
    - The script will display the uploaded image.
    - It will then detect and display the prominent colors in the image along with their RGB and HEX codes.

## Example Code

Here is the main script (`color_detector.py`):

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from google.colab import files

def upload_and_read_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        return img

def detect_colors(image, k=5):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    return colors

def display_colors(colors):
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.subplot(1, len(colors), i+1)
        plt.imshow([[color / 255]])
        plt.axis('off')
    plt.show()

def get_color_codes(colors):
    color_codes = []
    for color in colors:
        hex_code = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        color_codes.append(hex_code)
    return color_codes

def visualize_colors(image, colors):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    for i, color in enumerate(colors):
        plt.subplot(1, len(colors), i+1)
        plt.imshow([[color / 255]])
        plt.axis('off')
        plt.title(color_codes[i])
    plt.show()

if __name__ == "__main__":
    image = upload_and_read_image()
    detected_colors = detect_colors(image)
    display_colors(detected_colors)
    color_codes = get_color_codes(detected_colors)
    print("Detected Color Codes:", color_codes)
    visualize_colors(image, detected_colors)
```

## Contact

- **Name:** Zeeshan Ahmad
- **Email:** [ziishanahmad@gmail.com](mailto:ziishanahmad@gmail.com)
- **GitHub:** [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn:** [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)

Follow these steps to detect and visualize the colors in your images. This tool is useful for extracting color palettes and understanding the color composition of images.
