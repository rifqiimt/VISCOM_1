import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Perbandingan CLAHE Standar dan Adaptive CLAHE")

# Upload file gambar
uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Adaptive grid function
def get_adaptive_tile_size(mean_val):
    if mean_val < 100:
        return (4, 4)
    elif mean_val > 150:
        return (16, 16)
    else:
        return (8, 8)

# Jika file diupload
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clip_limits = [1.0, 2.0, 3.0]
    mean_val = np.mean(gray)

    fig, axes = plt.subplots(2, len(clip_limits), figsize=(15, 6))
    fig.suptitle('CLAHE Standar vs Adaptive (TileGridSize Dinamis)', fontsize=14)

    for idx, clip in enumerate(clip_limits):
        # CLAHE standar
        clahe_std = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        result_std = clahe_std.apply(gray)

        axes[0, idx].imshow(result_std, cmap='gray')
        axes[0, idx].set_title(f'CLAHE\nClipLimit={clip}, Grid=(8,8)')
        axes[0, idx].axis('off')

        # CLAHE adaptif
        adaptive_tile = get_adaptive_tile_size(mean_val)
        clahe_adaptive = cv2.createCLAHE(clipLimit=clip, tileGridSize=adaptive_tile)
        result_adaptive = clahe_adaptive.apply(gray)

        axes[1, idx].imshow(result_adaptive, cmap='gray')
        axes[1, idx].set_title(f'Adaptive CLAHE\nClipLimit={clip}, Grid={adaptive_tile}')
        axes[1, idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig)
