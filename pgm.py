import pywt
import numpy as np
import cv2

def convert_to_yuv(image):
    """Convert RGB image to YUV format."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

def apply_dwt(channel, wavelet='haar'):
    """Apply Discrete Wavelet Transform to a single channel."""
    coeffs = pywt.wavedec2(channel, wavelet)
    return coeffs

def inverse_dwt(coeffs, wavelet='haar'):
    """Inverse DWT to reconstruct the image channel."""
    return pywt.waverec2(coeffs, wavelet)

def sort_coeffs(coeffs):
    """Flatten and sort coefficients from large to small."""
    flat_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):
            for sub_c in c:
                flat_coeffs.extend(sub_c.flatten())
        else:
            flat_coeffs.extend(c.flatten())
    
    return np.sort(np.abs(np.array(flat_coeffs)))[::-1]

def cwdr_compress(channel, bits):
    """Compress a single channel using CWDR."""
    coeffs = apply_dwt(channel)
    sorted_coeffs = sort_coeffs(coeffs)

    N = np.ceil(np.log2(np.max(np.abs(sorted_coeffs)))).astype(int)
    threshold = 2 ** N
    n = 1

    compressed_coeffs = [[c.copy() for c in level] for level in coeffs]

    while bits > 0:
        for i, level in enumerate(compressed_coeffs):
            for j, sub_band in enumerate(level):
                mask = np.abs(sub_band) >= threshold
                sub_band[~mask] = 0

        n += 1
        threshold /= 2
        bits -= 1

    return compressed_coeffs

def compress_image(image, bits):
    """Compress an image using the CWDR algorithm."""
    yuv_image = convert_to_yuv(image)
    channels = cv2.split(yuv_image)

    compressed_channels = [cwdr_compress(channel, bits) for channel in channels]
    reconstructed_channels = [inverse_dwt(c) for c in compressed_channels]

    compressed_image = cv2.merge(reconstructed_channels).astype(np.uint8)
    return compressed_image

def main():
    # Load the image
    image = cv2.imread('ID_0088_AGE_0067_CONTRAST_0_CT.png')

    # Check if the image loaded successfully
    if image is None:
        print("Error: Could not load the image. Please check the path.")
        return

    bits = 8
    compressed_image = compress_image(image, bits)

    cv2.imwrite('compressed_image.jpg', compressed_image)
    print("Compression completed and saved as 'compressed_image.jpg'")

if __name__ == "__main__":
    main()
