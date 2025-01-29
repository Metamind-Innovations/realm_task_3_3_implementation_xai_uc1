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


# LIME Explanations
def generate_lime_explanations(model, img_slice, lung_mask, filename, output_dir):
    explainer = lime_image.LimeImageExplainer(random_state=42)

    # Normalize and create RGB image
    img_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
    img_rgb = np.stack([img_normalized] * 3, axis=-1)

    # Prediction function constrained to lung area
    def batch_predict(images):
        grayscale = images[..., 0].reshape(-1, 512, 512, 1)
        masked_images = grayscale * lung_mask[..., np.newaxis]
        predictions = model.predict(masked_images)
        return np.mean(predictions, axis=(1, 2))

    explanation = explainer.explain_instance(
        img_rgb,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=200
    )

    temp, mask_lime = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=5,
        hide_rest=False
    )

    # Calculate metrics
    intersection = np.sum(mask_lime * lung_mask)
    union = np.sum(mask_lime) + np.sum(lung_mask) - intersection
    metrics = {
        "confidence": np.max(explanation.score),
        "iou": intersection / (union + 1e-8),
        "recall": intersection / (np.sum(lung_mask) + 1e-8),
        "precision": intersection / (np.sum(mask_lime) + 1e-8),
        "f1_score": 2 * (intersection / (np.sum(lung_mask) + 1e-8)) * (intersection / (np.sum(mask_lime) + 1e-8)) /
                    ((intersection / (np.sum(lung_mask) + 1e-8)) + (intersection / (np.sum(mask_lime) + 1e-8)) + 1e-8),
        "coverage": intersection / (np.sum(lung_mask) + 1e-8)
    }

    # Save visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='bone')
    plt.title('Original Slice')
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp, mask_lime))
    plt.title('LIME Explanation')

    fig_path = f"{output_dir}/lime_explanation_{filename}.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return metrics, fig_path


# GRAD-CAM Implementation
class GradCAM:
    def __init__(self, model, layer_name="conv2d_23"):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )

    def compute_heatmap(self, img_array, eps=1e-8):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + eps)

        return heatmap

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4):
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * alpha + original_img * (1 - alpha)
        return np.clip(superimposed_img, 0, 255).astype(np.uint8)

    def explain(self, img_slice, original_img, output_dir, filename):
        img_array = np.expand_dims(img_slice, axis=0)[..., np.newaxis]
        heatmap = self.compute_heatmap(img_array)

        # Convert original image to RGB and normalize
        original_normalized = ((original_img - original_img.min()) /
                               (original_img.max() - original_img.min()) * 255).astype(np.uint8)
        original_rgb = np.stack([original_normalized] * 3, axis=-1)

        superimposed = self.overlay_heatmap(heatmap, original_rgb)

        # Save visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='bone')
        plt.title('Original Slice')
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed)
        plt.title('GRAD-CAM Explanation')

        fig_path = f"{output_dir}/gradcam_{filename}.png"
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        return fig_path


# Report Generation (common for both methods)
def generate_xai_report(method_name, fig_paths, metrics, output_dir):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    import time

    pdf_path = f"{output_dir}/{method_name}_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title Page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, f"{method_name} Explanations Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.showPage()

    # Content Pages
    for fig_path, metric in zip(fig_paths, metrics):
        img_reader = ImageReader(fig_path)
        img_width, img_height = img_reader.getSize()
        aspect = img_height / img_width
        c.drawImage(fig_path, 50, 400, width=500, height=500 * aspect)

        c.setFont("Helvetica", 10)
        y_pos = 380
        for key, value in metric.items():
            if isinstance(value, float):
                c.drawString(50, y_pos, f"{key}: {value:.2f}")
            else:
                c.drawString(50, y_pos, f"{key}: {value}")
            y_pos -= 20

        c.showPage()

    c.save()
    return pdf_path
