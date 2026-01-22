import cv2
import numpy as np
from pathlib import Path


# =========================
# Image loading
# =========================

def load_image(image_path):
    """
    Load an image from disk.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Failed to load image. Check the file path.")
    return image


# =========================
# Preprocessing steps
# =========================

def to_grayscale(image):
    """
    Convert a color image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray_image):
    """
    Improve contrast using CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def otsu_binarization(gray_image):
    """
    Apply Otsu thresholding (global threshold).
    """
    _, binary = cv2.threshold(
        gray_image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def deskew_image(binary_image):
    """
    Automatically correct image rotation using text orientation.
    """
    inverted = cv2.bitwise_not(binary_image)
    coords = np.column_stack(np.where(inverted > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_image.shape
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        binary_image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed, angle


def denoise_image(image):
    """
    Remove noise while preserving text edges.
    """
    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=30,
        templateWindowSize=7,
        searchWindowSize=21
    )


def save_processed_image(image, output_path):
    """
    Save processed image to disk.
    """
    cv2.imwrite(str(output_path), image)


def show_image(image, title="Image"):
    """
    Display an image.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# MAIN PIPELINE
# =========================

if __name__ == "__main__":
    img_path = Path("data/raw/test.jpg")
    output_path = Path("data/processed/test_processed.jpg")

    # 1. Load image
    original = load_image(img_path)

    # 2. Grayscale + contrast
    gray = to_grayscale(original)
    enhanced = enhance_contrast(gray)

    # 3. Binarization (Otsu)
    binary = otsu_binarization(enhanced)

    # 4. Deskew
    deskewed, angle = deskew_image(binary)
    print(f"Detected rotation angle: {angle:.2f} degrees")

    # 5. Denoise
    clean = denoise_image(deskewed)

    # 6. Save result
    save_processed_image(clean, output_path)
    print(f"Processed image saved to: {output_path}")

    # 7. Display
    show_image(original, "Original")
    show_image(clean, "Final OCR-Ready Image")
