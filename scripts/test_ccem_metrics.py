import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XAI_DIR = os.path.join(PROJECT_ROOT, "src", "xai")

for path in (PROJECT_ROOT, XAI_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from CCEM.ccem_core import (
    adaptive_reliability_ccem,
    calculate_advanced_metrics,
    calculate_extended_metrics,
    distance_pointing_game,
    component_filter_fusion,
    generate_ccem,
    soft_pointing_game,
    topk_pointing_game,
    saliency_precision_at_k,
)


class DummySeverityModel(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3), keepdim=True)


def _image(height=8, width=8):
    return torch.linspace(0.0, 1.0, steps=3 * height * width, dtype=torch.float32).view(1, 3, height, width)


def _model():
    return DummySeverityModel().eval()


def test_all_zero_heatmap_returns_no_localization_and_undefined_faithfulness():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[:2, :2] = 1.0

    ebpg, pg, sparsity, deletion, insertion, oa = calculate_advanced_metrics(
        _model(), _image(), heatmap, mask
    )

    assert ebpg == 0.0
    assert pg == 0.0
    assert sparsity == 0.0
    assert np.isnan(deletion)
    assert np.isnan(insertion)
    assert np.isnan(oa)


def test_heatmap_entirely_inside_mask_scores_localization():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[2:4, 2:4] = 2.0
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:4, 2:4] = 1.0

    ebpg, pg, sparsity, *_ = calculate_advanced_metrics(_model(), _image(), heatmap, mask)

    assert np.isclose(ebpg, 1.0)
    assert pg == 1.0
    assert 0.0 <= sparsity <= 1.0


def test_heatmap_entirely_outside_mask_scores_no_localization():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[5:7, 5:7] = 1.0
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[1:3, 1:3] = 1.0

    ebpg, pg, *_ = calculate_advanced_metrics(_model(), _image(), heatmap, mask)

    assert np.isclose(ebpg, 0.0)
    assert pg == 0.0


def test_sparsity_is_bounded_for_random_heatmaps():
    rng = np.random.default_rng(123)
    mask = np.ones((8, 8), dtype=np.float32)

    for _ in range(20):
        heatmap = rng.normal(size=(8, 8)).astype(np.float32)
        _, _, sparsity, *_ = calculate_advanced_metrics(_model(), _image(), heatmap, mask)
        assert 0.0 <= sparsity <= 1.0


def test_heatmap_and_mask_are_resized_to_model_input_shape():
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[1:3, 1:3] = 1.0
    mask = np.zeros((5, 5), dtype=np.float32)
    mask[1:4, 1:4] = 1.0

    metrics = calculate_advanced_metrics(_model(), _image(8, 8), heatmap, mask)

    assert np.isfinite(metrics[0])
    assert metrics[1] in (0.0, 1.0)
    assert 0.0 <= metrics[2] <= 1.0


def test_deletion_insertion_and_oa_ranges_for_nondegenerate_response():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[2:6, 2:6] = 1.0
    mask = np.ones((8, 8), dtype=np.float32)

    _, _, _, deletion, insertion, oa = calculate_advanced_metrics(
        _model(), _image(), heatmap, mask, n_steps=6
    )

    assert 0.0 <= deletion <= 1.0
    assert 0.0 <= insertion <= 1.0
    assert -1.0 <= oa <= 1.0


def test_metric_accepts_singleton_and_channel_heatmap_shapes():
    heatmap = np.zeros((1, 8, 8), dtype=np.float32)
    heatmap[:, 2:4, 2:4] = 1.0
    mask = np.zeros((8, 8, 1), dtype=np.float32)
    mask[2:4, 2:4, :] = 1.0

    ebpg, pg, sparsity, *_ = calculate_advanced_metrics(_model(), _image(), heatmap, mask)

    assert np.isclose(ebpg, 1.0)
    assert pg == 1.0
    assert 0.0 <= sparsity <= 1.0


def test_component_filter_falls_back_instead_of_erasing_nonzero_fusion():
    fused = np.zeros((16, 16), dtype=np.float32)
    fused[2:4, 2:4] = 0.6
    fused[10:12, 10:12] = 1.0

    filtered = component_filter_fusion(
        fused,
        gradcam=fused,
        adasise=fused,
        ig_smooth=fused,
        min_area=1000,
    )

    assert float(np.max(filtered)) > 1e-8
    assert float(np.sum(filtered)) > 1e-8


def test_new_ccem_modes_preserve_nonzero_ig_signal():
    rng = np.random.default_rng(7)
    gradcam = rng.random((12, 12), dtype=np.float32) * 0.25
    adasise = rng.random((12, 12), dtype=np.float32) * 0.50
    ig = np.zeros((12, 12), dtype=np.float32)
    ig[3:9, 3:9] = np.linspace(0.1, 1.0, 36, dtype=np.float32).reshape(6, 6)

    for mode in ("ig_only", "simple_average", "soft_agreement", "ig_anchored", "adaptive_reliability"):
        heatmap = generate_ccem(gradcam, adasise, ig, mode=mode)
        assert heatmap.shape == ig.shape
        assert float(np.max(heatmap)) > 1e-8
        assert float(np.sum(heatmap)) > 1e-8


def test_extended_metrics_all_zero_heatmap_localization_is_zero():
    heatmap = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:4, 2:4] = 1.0

    metrics = calculate_extended_metrics(_model(), _image(), heatmap, mask)

    assert metrics["EBPG"] == 0.0
    assert metrics["PG"] == 0.0
    assert metrics["SoftPG"] == 0.0
    assert metrics["TopKPG_1"] == 0.0
    assert metrics["Precision_1"] == 0.0
    assert metrics["DistancePG"] == 0.0


def test_extended_metrics_inside_mask_score_localization():
    heatmap = np.zeros((20, 20), dtype=np.float32)
    heatmap[5:10, 5:10] = 1.0
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:10, 5:10] = 1.0

    metrics = calculate_extended_metrics(_model(), _image(20, 20), heatmap, mask)

    assert np.isclose(metrics["EBPG"], 1.0)
    assert metrics["PG"] == 1.0
    assert np.isclose(metrics["SoftPG"], 1.0)
    assert metrics["Precision_1"] > 0.0


def test_soft_pointing_gives_credit_for_secondary_in_mask_response():
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[1, 1] = 1.0
    heatmap[8, 8] = 0.4
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[7:10, 7:10] = 1.0

    metrics = calculate_extended_metrics(_model(), _image(16, 16), heatmap, mask)

    assert metrics["PG"] == 0.0
    assert metrics["SoftPG"] > 0.0
    assert metrics["EBPG"] > 0.0


def test_distance_pointing_game_decays_with_distance():
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[15:18, 15:18] = 1.0

    inside = np.zeros((32, 32), dtype=np.float32)
    inside[16, 16] = 1.0
    near = np.zeros((32, 32), dtype=np.float32)
    near[16, 22] = 1.0
    far = np.zeros((32, 32), dtype=np.float32)
    far[1, 1] = 1.0

    inside_score = distance_pointing_game(inside, mask, sigma=5.0)
    near_score = distance_pointing_game(near, mask, sigma=5.0)
    far_score = distance_pointing_game(far, mask, sigma=5.0)

    assert np.isclose(inside_score, 1.0)
    assert inside_score > near_score > far_score


def test_adaptive_ccem_returns_debug_weights_and_nonzero_map():
    gradcam = np.zeros((12, 12), dtype=np.float32)
    adasise = np.zeros((12, 12), dtype=np.float32)
    adasise[2:6, 2:6] = 1.0
    ig = np.zeros((12, 12), dtype=np.float32)
    ig[6:10, 6:10] = 1.0

    fused, debug = adaptive_reliability_ccem(
        gradcam,
        adasise,
        ig,
        image_tensor=_image(12, 12),
        model=_model(),
        return_debug=True,
    )

    assert fused.shape == ig.shape
    assert float(np.max(fused)) > 1e-8
    assert set(debug["weights"]) == {"GradCAM++", "Ada-SISE", "IG_Smooth"}
    assert set(debug["reliability"]) == {"GradCAM++", "Ada-SISE", "IG_Smooth"}
    assert np.isclose(sum(debug["weights"].values()), 1.0)
    assert debug["reliability"]["GradCAM++"] < debug["reliability"]["Ada-SISE"]


def test_adaptive_soft_keep_percentile_preserves_nonzero_map():
    rng = np.random.default_rng(12)
    gradcam = rng.random((12, 12), dtype=np.float32)
    adasise = rng.random((12, 12), dtype=np.float32)
    ig = rng.random((12, 12), dtype=np.float32)

    fused = generate_ccem(
        gradcam,
        adasise,
        ig,
        mode="adaptive_reliability",
        adaptive_soft_keep_percentile=60,
    )

    assert fused.shape == ig.shape
    assert float(np.max(fused)) > 1e-8


def _run_without_pytest():
    tests = [
        test_all_zero_heatmap_returns_no_localization_and_undefined_faithfulness,
        test_heatmap_entirely_inside_mask_scores_localization,
        test_heatmap_entirely_outside_mask_scores_no_localization,
        test_sparsity_is_bounded_for_random_heatmaps,
        test_heatmap_and_mask_are_resized_to_model_input_shape,
        test_deletion_insertion_and_oa_ranges_for_nondegenerate_response,
        test_metric_accepts_singleton_and_channel_heatmap_shapes,
        test_component_filter_falls_back_instead_of_erasing_nonzero_fusion,
        test_new_ccem_modes_preserve_nonzero_ig_signal,
        test_extended_metrics_all_zero_heatmap_localization_is_zero,
        test_extended_metrics_inside_mask_score_localization,
        test_soft_pointing_gives_credit_for_secondary_in_mask_response,
        test_distance_pointing_game_decays_with_distance,
        test_adaptive_ccem_returns_debug_weights_and_nonzero_map,
        test_adaptive_soft_keep_percentile_preserves_nonzero_map,
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} CCEM metric tests.")


if __name__ == "__main__":
    _run_without_pytest()
