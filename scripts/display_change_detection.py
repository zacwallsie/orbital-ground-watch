import numpy as np
from utils.display_utils import plot_imgs


x_val = np.load("net_based/prediction_output/x.npy")
y_val = np.load("net_based/prediction_output/y.npy")

y_pred = np.load("net_based/prediction_output/w_normalisation/predicted_masks.npy")

x_val = np.transpose(x_val, (0, 2, 3, 1))
y_val = np.transpose(y_val, (0, 2, 3, 1))

r, g, b = [2.9, 2.6, 2.3]

before_arrays = np.stack(
    [x_val[:, ..., 3] * r, x_val[:, ..., 2] * g, x_val[:, ..., 1] * b], 3
)
after_arrays = np.stack(
    [x_val[:, ..., 9] * r, x_val[:, ..., 8] * g, x_val[:, ..., 7] * b], 3
)

print(x_val.shape)
print(y_val.shape)


plot_imgs(
    before_imgs=before_arrays,
    after_imgs=after_arrays,
    mask_imgs=y_val,
    pred_imgs=y_pred,
    nm_img_to_plot=5,
    figsize=3,
)
