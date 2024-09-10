import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def split_and_transpose_image_pairs(input_array):
    # Check if the input array has the correct number of channels
    if input_array.shape[-1] != 12:
        raise ValueError("Input array must have 12 channels")

    # Split the array along the last axis (channels)
    array1 = input_array[..., :6]
    array2 = input_array[..., 6:]

    # Transpose the arrays from (N, H, W, C) to (N, C, H, W)
    array1 = np.transpose(array1, (0, 3, 1, 2))
    array2 = np.transpose(array2, (0, 3, 1, 2))

    return array1, array2


def visualize_split(original, split1, split2, image_index=0):
    """
    Visualize the original image and the split results for a given image index.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original[image_index, :, :, :3])  # Show first 3 channels of original
    axes[0].set_title("Original (First 3 channels)")
    axes[0].axis("off")

    # Split image 1 (transposed)
    axes[1].imshow(np.transpose(split1[image_index, :3, :, :], (1, 2, 0)))
    axes[1].set_title("Split 1 (First 3 channels)")
    axes[1].axis("off")

    # Split image 2 (transposed)
    axes[2].imshow(np.transpose(split2[image_index, :3, :, :], (1, 2, 0)))
    axes[2].set_title("Split 2 (First 3 channels)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main(input_file, output_dir):
    # Load the input NumPy array
    input_array = np.load(input_file)

    # Split and transpose the image pairs
    result1, result2 = split_and_transpose_image_pairs(input_array)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the resulting NumPy arrays
    np.save(os.path.join(output_dir, "split1_transposed.npy"), result1)
    np.save(os.path.join(output_dir, "split2_transposed.npy"), result2)

    print("Input shape:", input_array.shape)
    print("Output 1 shape:", result1.shape)
    print("Output 2 shape:", result2.shape)
    print(f"Results saved in {output_dir}")

    # Visualize the first image (index 0)
    visualize_split(input_array, result1, result2, image_index=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split and transpose image pairs from a NumPy dataset."
    )
    parser.add_argument(
        "input_file", help="Path to the input NumPy dataset file (.npy)"
    )
    parser.add_argument(
        "output_dir", help="Directory to save the resulting NumPy datasets"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
