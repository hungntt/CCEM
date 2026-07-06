import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus


class RegressionOutputTarget:
    """Target for regression models: scales the scalar model output."""

    def __init__(self, sign=1.0):
        self.sign = float(sign)

    def __call__(self, model_output):
        return self.sign * model_output.reshape(-1)[0]


def _usable_heatmap(heatmap, eps=1e-8):
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)
    return bool(hm.size and float(np.max(hm)) > eps and float(np.sum(hm)) > eps)


class CompactGradCAMPlusPlus:
    """
    Core wrapper for Grad-CAM++ to maintain a consistent API across all explainers.
    It automatically determines the optimal target layer (CBAM or final Conv block)
    and generates the raw heatmap.
    """
    def __init__(self, model, use_cbam=False):
        self.model = model
        self.use_cbam = use_cbam

        # Determine the target layer dynamically based on the architecture
        if self.use_cbam and hasattr(model, 'cbam'):
            self.target_layers = [model.cbam]
        elif hasattr(model.encoder, 'conv_head'):
            self.target_layers = [model.encoder.conv_head]
        else:
            # Fallback for other timm models
            self.target_layers = [list(model.encoder.children())[-2]]

        self.cam_explainer = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)
        self.gradcam_fallback = GradCAM(model=self.model, target_layers=self.target_layers)

        self.last_fallback_reason = "none"

    def _run_cam(self, cam, input_tensor, targets, eigen_smooth=False):
        raw_heatmap = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
        )[0, :]
        return np.asarray(raw_heatmap, dtype=np.float32)

    def generate_heatmap(self, input_tensor, target_class=0):
        """
        Generates the raw Grad-CAM++ heatmap, using the regression output as the
        attribution target. Falls back through eigen-smoothed Grad-CAM++, vanilla
        Grad-CAM, and a sign-flipped regression target if earlier stages degenerate
        to an empty/all-zero map.

        Returns:
            raw_heatmap (numpy.ndarray): The 2D heatmap array.
            target_class (int): The class index used for attribution (always 0).
        """
        del target_class

        self.last_fallback_reason = "none"
        targets = [RegressionOutputTarget(sign=1.0)]

        raw_heatmap = self._run_cam(
            self.cam_explainer,
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=False,
        )

        if not _usable_heatmap(raw_heatmap):
            self.last_fallback_reason = "gradcampp_eigen_smooth"
            raw_heatmap = self._run_cam(
                self.cam_explainer,
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=True,
            )

        if not _usable_heatmap(raw_heatmap):
            self.last_fallback_reason = "vanilla_gradcam"
            raw_heatmap = self._run_cam(
                self.gradcam_fallback,
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=False,
            )

        if not _usable_heatmap(raw_heatmap):
            self.last_fallback_reason = "negative_regression_target"
            raw_heatmap = self._run_cam(
                self.cam_explainer,
                input_tensor=input_tensor,
                targets=[RegressionOutputTarget(sign=-1.0)],
                eigen_smooth=False,
            )

        return raw_heatmap, 0

    def get_target_layer_name(self):
        """Returns the name of the layer being targeted for gradients."""
        return self.target_layers[0].__class__.__name__
