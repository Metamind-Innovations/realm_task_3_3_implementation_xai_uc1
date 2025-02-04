import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import time

# Visualization constants
CMAP_BONE = 'bone'
FONT_NAME = 'Helvetica'
FONT_BOLD = 'Helvetica-Bold'
GRADCAM_LAYER_NAMES = ['conv2d_22', 'conv2d_19', 'conv2d_16', 'conv2d_13']  # Optimal layers from U-Net decoder


def generate_lime_explanations(model, img_slice, lung_mask, filename, output_dir):
    """Generate LIME explanations for a single slice."""
    # Normalize image
    img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    img_rgb = np.stack([img_normalized] * 3, axis=-1)

    # Create explainer and prediction function
    explainer = lime_image.LimeImageExplainer(random_state=42)
    batch_predict = lambda images: _lime_predict(model, images, lung_mask)

    # Generate explanation
    explanation = explainer.explain_instance(
        img_rgb,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )

    # Process results
    image, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
    )

    # Calculate metrics
    metrics = _calculate_metrics(mask, lung_mask, explanation)

    # Save visualizations
    lime_path = _save_lime_visualization(image, mask, output_dir, filename)

    return metrics, lime_path


def _lime_predict(model, images, lung_mask):
    """Helper function for LIME predictions."""
    grayscale = images[..., 0].reshape(-1, 512, 512, 1)
    masked = grayscale * lung_mask[..., np.newaxis]
    return np.mean(model.predict(masked), axis=(1, 2))


def _calculate_metrics(mask, lung_mask, explanation):
    """Calculate XAI metrics."""
    intersection = np.sum(mask * lung_mask)
    union = np.sum(mask) + np.sum(lung_mask) - intersection

    return {
        "confidence": np.max(explanation.score),
        "iou": intersection / (union + 1e-8),
        "recall": intersection / (np.sum(lung_mask) + 1e-8),
        "precision": intersection / (np.sum(mask) + 1e-8),
        "f1_score": 2 * (intersection / (np.sum(lung_mask) + 1e-8)) *
                    (intersection / (np.sum(mask) + 1e-8)) /
                    ((intersection / (np.sum(lung_mask) + 1e-8)) +
                     (intersection / (np.sum(mask) + 1e-8)) + 1e-8)
    }

def _save_lime_visualization(lime_image, mask, output_dir, filename):
    """Save LIME explanation visualization."""
    path = f"{output_dir}/lime_{filename}.png"
    plt.figure(figsize=(5,5))
    plt.imshow(mark_boundaries(lime_image, mask))
    plt.title('LIME Explanation')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()
    return path


def generate_gradcam_explanation(model, input_slice, output_dir, filename, lung_mask=None, layer_name=None):
    """Generate Grad-CAM explanation with automatic layer selection."""
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

    # Prepare input
    input_array = np.expand_dims(input_slice, axis=0)[..., np.newaxis].astype(np.float32)

    # Compute heatmap
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_array)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generate and process heatmap
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


def _normalize_to_rgb(img_slice):
    """Convert slice to normalized RGB format."""
    normalized = ((img_slice - img_slice.min()) /
                  (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)


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
    plt.imshow(input_slice, cmap=CMAP_BONE)  # Grayscale background
    plt.imshow(heatmap, cmap='jet', alpha=0.4)  # Transparent heatmap overlay
    plt.title('GRAD-CAM')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()
    return path


# TODO: Remove metrics
def generate_combined_report(original_paths, lime_paths, gradcam_paths, metrics, output_dir):
    """Generate combined XAI report with all explanations in one PDF."""
    pdf_path = f"{output_dir}/XAI_Combined_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title Page
    c.setFont(FONT_BOLD, 18)
    c.drawString(50, 750, "Combined XAI Report")
    c.setFont(FONT_NAME, 12)
    c.drawString(50, 720, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.showPage()

    for orig_path, lime_path, gradcam_path, metric in zip(original_paths, lime_paths, gradcam_paths, metrics):
        # Image positions
        y_pos = 550
        img_width = 180

        # Only draw available explanations
        current_x = 50
        if orig_path:
            c.drawImage(ImageReader(orig_path), current_x, y_pos, width=img_width, height=img_width)
            current_x += 150

        if lime_path:
            c.drawImage(ImageReader(lime_path), current_x, y_pos, width=img_width, height=img_width)
            current_x += 150

        if gradcam_path:
            c.drawImage(ImageReader(gradcam_path), current_x, y_pos, width=img_width, height=img_width)

        # Metrics below images
        c.setFont(FONT_NAME, 12)
        y_metrics = y_pos - 60
        for key, value in metric.items():
            c.drawString(50, y_metrics, f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
            y_metrics -= 20

        c.showPage()

    c.save()