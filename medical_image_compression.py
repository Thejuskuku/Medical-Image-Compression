import cv2
import math
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

# Define color scheme
PRIMARY_COLOR = '#0275d8'
SECONDARY_COLOR = '#f0ad4e'
LIGHT_COLOR = '#ffffff'
DARK_COLOR = '#333333'

# Define font styles
HEADER_FONT = ('Arial', 18, 'bold')
BUTTON_FONT = ('Arial', 12)

def compress_image():
    # Prompt user to select image file
    filename = filedialog.askopenfilename(title="Select an image file to compress")

    # Check if user canceled file selection
    if not filename:
        return

    try:
        # Load image
        img = cv2.imread(filename)

        # Resize image
        img = cv2.resize(img, (512, 512))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Huffman coding
        huffmanen = cv2.imencode('.png', gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]

        # Resize image
        # huffmanen = cv2.resize(huffmanen, (512, 512))

        # Save compressed image
        with open('compressed.png', 'wb') as f:
            f.write(huffmanen)

        # Display original and compressed images
        cv2.imshow('Original Image', img)
        cv2.imshow('Compressed Image', cv2.imread('compressed.png'.replace('\\', '/')))

        # Calculate compression ratio
        orig_size = img.shape[0] * img.shape[1] * img.shape[2]
        compressed_size = len(huffmanen)
        ratio = orig_size / compressed_size
        ratio_label.config(text=f'Compression Ratio: {ratio:.2f}')

        # Calculate PSNR
        orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        compressed_img = cv2.imread('compressed.png', cv2.IMREAD_GRAYSCALE)
        mse = np.mean((orig_img - compressed_img) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        psnr_label.config(text=f'PSNR: {psnr:.2f}')

    except Exception as e:
        # Display error message if image processing fails
        error_msg = f"Error: {e}"
        error_label.config(text=error_msg)

    finally:
        # Enable compress button and clear error message
        compress_button.config(state=NORMAL)
        error_label.config(text='')

def decompress_image():
    # Prompt user to select compressed image file
    filename = filedialog.askopenfilename(title="Select a compressed image file to decompress")

    # Check if user canceled file selection
    if not filename:
        return

    try:
        # Load compressed image
        with open(filename, 'rb') as f:
            img_data = f.read()

        # Apply Huffman coding
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        huffmande = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize image
        # huffmande = cv2.resize(huffmande, (512, 512))

        # Save decompressed image
        cv2.imwrite('decompressed.png', huffmande)

        # Display decompressed image
        cv2.imshow('Decompressed Image', huffmande)

    except Exception as e:
        # Display error message if image processing fails
        error_msg = f"Error: {e}"
        error_label.config(text=error_msg)

    finally:
        # Enable decompress button and clear error message
        decompress_button.config(state=NORMAL)
        error_label.config(text='')

# Create GUI
root = tk.Tk()
root.title("Medical Image Compression")
root.geometry("500x300")

# Add header label
header_label = tk.Label(root, text="Medical Image Compression", font=HEADER_FONT, fg=PRIMARY_COLOR)
header_label.pack(pady=20)

# Add compress button
compress_button = tk.Button(root, text="Compress Image", font=BUTTON_FONT, bg=PRIMARY_COLOR, fg=LIGHT_COLOR, command=compress_image)
compress_button.pack(pady=10)

# Add decompress button
decompress_button = tk.Button(root, text="Decompress Image", font=BUTTON_FONT, bg=SECONDARY_COLOR, fg=LIGHT_COLOR, command=decompress_image)
decompress_button.pack(pady=10)

# Add compression ratio label
ratio_label = tk.Label(root, text="Compression Ratio:")
ratio_label.pack(pady=10)

# Add PSNR label
psnr_label = tk.Label(root, text="PSNR: ")
psnr_label.pack(pady=10)

# Add error label
error_label = tk.Label(root, text="", fg='red')
error_label.pack(pady=10)

# Start GUI
root.mainloop()