import numpy as np
from PIL import Image

# Step 1: Load low-light and bright-light images (grayscale)
low_img = np.array(Image.open("low_light.jpg").convert("L"))
bright_img = np.array(Image.open("bright_light.jpg").convert("L"))

# Step 2: Function to get raw (unscaled) bit-planes (0 or 1)
def bit_planes_raw(img):
    return np.array([(img >> i) & 1 for i in range(8)], dtype=np.uint8)

low_planes_raw = bit_planes_raw(low_img)
bright_planes_raw = bit_planes_raw(bright_img)

# Step 3: Reconstruct image using 3 lowest planes of each (raw bits)
def reconstruct_lowest_3(raw_planes):
    return (
        (raw_planes[0]) +
        (raw_planes[1] << 1) +
        (raw_planes[2] << 2)
    ).astype(np.uint8)

low_recon = reconstruct_lowest_3(low_planes_raw)
bright_recon = reconstruct_lowest_3(bright_planes_raw)

# Step 4: UNION of the two reconstructions (max grayscale)
union_recon = np.maximum(low_recon, bright_recon)

# Step 5: Differences (convert to int16 to avoid overflow)
diff_low = np.abs(low_img.astype(np.int16) - union_recon.astype(np.int16)).astype(np.uint8)
diff_bright = np.abs(bright_img.astype(np.int16) - union_recon.astype(np.int16)).astype(np.uint8)

# Step 6: Save outputs
Image.fromarray(low_recon).save("Low_Reconstructed.png")
Image.fromarray(bright_recon).save("Bright_Reconstructed.png")
Image.fromarray(union_recon).save("Union_Reconstructed.png")
Image.fromarray(diff_low).save("Low_Difference.png")
Image.fromarray(diff_bright).save("Bright_Difference.png")
