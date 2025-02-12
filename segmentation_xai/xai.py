import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Visualization constants
CMAP_BONE = 'bone'
GRADCAM_LAYER_NAMES = ['conv2d_22', 'conv2d_19', 'conv2d_16', 'conv2d_13']


def generate_gradcam_explanation(model, input_slice, output_dir, filename, lung_mask=None, layer_name=None):
    """Generate Grad-CAM visualization with automatic layer selection.

    Raises:
        ValueError: If no valid convolutional layer is found in model
    """
    # Auto-select layer if not specified
    if layer_name is None:
        for candidate_layer in GRADCAM_LAYER_NAMES:
            try:
                # Just check if layer exists, don't create full model yet
                model.get_layer(candidate_layer)
                layer_name = candidate_layer
                break
            except ValueError:
                continue
        if layer_name is None:
            raise ValueError("No valid Grad-CAM layer found in model")

    # Create gradient model - SINGLE CREATION POINT
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    # Prepare input tensor for Grad-CAM computation and calculate heatmap
    input_array = np.expand_dims(input_slice, axis=0)[..., np.newaxis].astype(np.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_array)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    heatmap = cv2.resize(heatmap, (input_slice.shape[1], input_slice.shape[0]))  # Resize to original dimensions

    if lung_mask is not None:
        heatmap *= lung_mask

    return _save_gradcam_visualization(input_slice, heatmap, output_dir, filename, layer_name)


def _compute_heatmap(grad_model, img_array, eps=1e-8):
    """Compute Grad-CAM heatmap using gradient model."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + eps)
    return heatmap


def _overlay_heatmap(heatmap, background, alpha=0.4):
    """Overlay heatmap on background image."""
    heatmap = cv2.resize(heatmap, (background.shape[1], background.shape[0]))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_colored, alpha, background, 1 - alpha, 0)


def _save_gradcam_visualization(input_slice, heatmap, output_dir, filename, layer_name):
    """Save Grad-CAM visualization without original slice."""
    path = f"{output_dir}/gradcam_{filename}_{layer_name}.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(input_slice, cmap=CMAP_BONE)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)  # Transparent heatmap overlay
    plt.title('GRAD-CAM')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()
    return path
