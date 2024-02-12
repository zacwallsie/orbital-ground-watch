import numpy as np
import matplotlib.pyplot as plt

# plot_imgs function taken directly from keras_unet/utils.py
# Also contains other potentially useful utilities Image Augmentation and splitting a large image into smaller chips
# https://github.com/karolzak/keras-unet

MASK_COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]


def plot_imgs(
        before_imgs,
        after_imgs,
        mask_imgs,
        pred_imgs=None,
        nm_img_to_plot=10,
        figsize=4,
        alpha=0.5,
        color="red"):
    """
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    Args:
        org_imgs (numpy.ndarray): Array of arrays representing a collection of original images.
        mask_imgs (numpy.ndarray): Array of arrays representing a collection of mask images (grayscale).
        pred_imgs (numpy.ndarray, optional): Array of arrays representing a collection of prediction masks images.. Defaults to None.
        nm_img_to_plot (int, optional): How many images to display. Takes first N images. Defaults to 10.
        figsize (int, optional): Matplotlib figsize. Defaults to 4.
        alpha (float, optional): Transparency for mask overlay on original image. Defaults to 0.5.
        color (str, optional): Color for mask overlay. Defaults to "red".
    """  # NOQA E501
    assert (color in MASK_COLORS)

    if nm_img_to_plot > before_imgs.shape[0]:
        nm_img_to_plot = before_imgs.shape[0]

    org_imgs_size = before_imgs.shape[1]

    if not (pred_imgs is None):
        cols = 5
    else:
        cols = 4

    im_id = 0

    for _ in range(0, before_imgs.shape[0], nm_img_to_plot):
        fig, axes = plt.subplots(
            nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False
        )
        axes[0, 0].set_title("before", fontsize=15)
        axes[0, 1].set_title("after", fontsize=15)
        axes[0, 2].set_title("ground truth", fontsize=15)

        if not (pred_imgs is None):
            axes[0, 3].set_title("prediction", fontsize=15)
            axes[0, 4].set_title("overlay", fontsize=15)
        else:
            axes[0, 3].set_title("overlay", fontsize=15)

        for m in range(0, nm_img_to_plot):

            axes[m, 0].imshow(before_imgs[im_id], cmap=_get_cmap(before_imgs))
            axes[m, 0].set_axis_off()
            axes[m, 1].imshow(after_imgs[im_id], cmap=_get_cmap(after_imgs))
            axes[m, 1].set_axis_off()
            axes[m, 2].imshow(mask_imgs[im_id], cmap=_get_cmap(mask_imgs))
            axes[m, 2].set_axis_off()
            if not (pred_imgs is None):
                axes[m, 3].imshow(pred_imgs[im_id], cmap=_get_cmap(pred_imgs))
                axes[m, 3].set_axis_off()
                axes[m, 4].imshow(after_imgs[im_id], cmap=_get_cmap(after_imgs))
                axes[m, 4].imshow(
                    _mask_to_rgba(
                        _zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                        color=color,
                    ),
                    cmap=_get_cmap(pred_imgs),
                    alpha=alpha,
                )
                axes[m, 4].set_axis_off()
            else:
                axes[m, 3].imshow(after_imgs[im_id], cmap=_get_cmap(after_imgs))
                axes[m, 3].imshow(
                    _mask_to_rgba(
                        _zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size),
                        color=color,
                    ),
                    cmap=_get_cmap(mask_imgs),
                    alpha=alpha,
                )
                axes[m, 3].set_axis_off()
            im_id += 1

        plt.show()


def _mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.

    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".

    Returns:
        numpy.ndarray: [description]
    """
    assert (color in MASK_COLORS)
    assert (mask.ndim == 3 or mask.ndim == 2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)


def _zero_pad_mask(mask, desired_size):
    """[summary]

    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]

    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def _reshape_arr(arr):
    """[summary]

    Args:
        arr (numpy.ndarray): [description]

    Returns:
        numpy.ndarray: [description]
    """
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def _get_cmap(arr):
    """[summary]

    Args:
        arr (numpy.ndarray): [description]

    Returns:
        string: [description]
    """
    if arr.ndim == 3:
        return "gray"
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return "jet"
        elif arr.shape[3] == 1:
            return "gray"