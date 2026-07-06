import cv2
import torch
import numpy as np
from scipy.stats import spearmanr

COMPONENT_FILTER_FALLBACK_COUNT = 0

# ==========================================
# MATHEMATICAL UTILITIES
# ==========================================
def ensure_2d_float_map(heatmap):
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.squeeze(hm)

    if hm.ndim == 3:
        if hm.shape[-1] in (1, 3, 4):
            hm = hm.mean(axis=-1)
        elif hm.shape[0] in (1, 3, 4):
            hm = hm.mean(axis=0)
        else:
            hm = hm.mean(axis=-1)

    if hm.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap after squeezing, got shape {hm.shape}.")

    return hm.astype(np.float32)

def min_max_normalize(heatmap):
    hm = ensure_2d_float_map(heatmap)
    hm = hm - np.min(hm)
    denom = np.max(hm)
    if denom < 1e-8:
        return np.zeros_like(hm, dtype=np.float32)
    return (hm / (denom + 1e-8)).astype(np.float32)

def percentile_normalize(hm, p_low=1, p_high=99):
    hm = ensure_2d_float_map(hm)

    lo, hi = np.percentile(hm, [p_low, p_high])
    if hi - lo < 1e-8:
        return np.zeros_like(hm, dtype=np.float32)

    hm = np.clip(hm, lo, hi)
    hm = (hm - lo) / (hi - lo + 1e-8)
    return hm.astype(np.float32)

def normalize_heatmap(heatmap, lower_percentile=1.0, upper_percentile=99.0):
    return percentile_normalize(
        heatmap,
        p_low=lower_percentile,
        p_high=upper_percentile,
    )

def robust_normalize(heatmap, p_low=1.0, p_high=99.0, eps=1e-8):
    """
    Percentile normalize, but compute percentiles over the positive pixels only.
    Sparse saliency maps (lesion-sized activations on a mostly-zero background)
    can have <1% nonzero pixels, so percentiles taken over the full flattened
    map collapse to zero and erase a genuinely nonzero map.
    """
    hm = ensure_2d_float_map(heatmap)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0).astype(np.float32)

    raw_max = float(np.max(hm)) if hm.size else 0.0
    raw_min = float(np.min(hm)) if hm.size else 0.0
    if raw_max - raw_min <= eps:
        return np.zeros_like(hm, dtype=np.float32)

    positive = hm[hm > eps]
    if positive.size >= 10:
        lo, hi = np.percentile(positive, [p_low, p_high])
    else:
        lo, hi = np.percentile(hm, [p_low, p_high])

    if hi - lo > eps:
        clipped = np.clip(hm, lo, hi)
        out = ((clipped - lo) / (hi - lo + eps)).astype(np.float32)
        out = np.maximum(out, 0.0)
        if float(np.max(out)) > eps and float(np.sum(out)) > eps:
            return out

    return min_max_normalize(hm)

def rank_normalize(hm):
    hm = ensure_2d_float_map(hm)
    if hm.size == 0 or float(np.max(hm) - np.min(hm)) < 1e-8:
        return np.zeros_like(hm, dtype=np.float32)

    flat = hm.reshape(-1)
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return ranks.reshape(hm.shape).astype(np.float32)

def resize_to_match(hm, target_shape, interpolation=cv2.INTER_LINEAR):
    hm = ensure_2d_float_map(hm)
    if hm.shape[:2] == target_shape[:2]:
        return hm.astype(np.float32)

    h, w = target_shape[:2]
    return cv2.resize(
        hm.astype(np.float32),
        (w, h),
        interpolation=interpolation,
    ).astype(np.float32)

def resize_like(heatmap, reference_shape):
    return resize_to_match(heatmap, reference_shape, interpolation=cv2.INTER_CUBIC)

def make_retina_mask(image):
    img = np.asarray(image)

    if img.ndim == 3:
        if np.issubdtype(img.dtype, np.floating) and img.max(initial=0) <= 1.0:
            img = np.clip(img * 255.0, 0, 255)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        if np.issubdtype(img.dtype, np.floating) and img.max(initial=0) <= 1.0:
            img = np.clip(img * 255.0, 0, 255)
        gray = img.astype(np.uint8)

    mask = (gray > 5).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.float32)

def connected_component_filter(
    heatmap,
    min_area=4,
    max_area=2200,
    max_aspect_ratio=7.0,
    component_percentile=75.0,
):
    hm = min_max_normalize(heatmap)
    positive = hm[hm > 0]

    if positive.size == 0:
        return hm.astype(np.float32)

    threshold = np.percentile(positive, component_percentile)
    binary = (hm >= threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    keep = np.zeros_like(binary, dtype=np.float32)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        if area < min_area or area > max_area:
            continue

        aspect = max(bw / (bh + 1e-8), bh / (bw + 1e-8))
        if aspect > max_aspect_ratio and area > 40:
            continue

        keep[labels == label] = 1.0

    if np.sum(keep) == 0:
        return hm.astype(np.float32)

    return min_max_normalize(hm * keep).astype(np.float32)

def component_filter_fusion(
    fused,
    gradcam,
    adasise,
    ig_smooth,
    min_area=20,
    max_area_ratio=0.10,
    top_k_components=8,
    threshold_percentile=88,
):
    global COMPONENT_FILTER_FALLBACK_COUNT

    fused = percentile_normalize(fused)
    gradcam = resize_to_match(percentile_normalize(gradcam), fused.shape)
    adasise = resize_to_match(percentile_normalize(adasise), fused.shape)
    ig_smooth = resize_to_match(percentile_normalize(ig_smooth), fused.shape)

    h, w = fused.shape
    max_area = int(max_area_ratio * h * w)
    threshold = np.percentile(fused, threshold_percentile)
    binary = (fused >= threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components = []
    for label in range(1, num_labels):
        region = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue

        fused_score = float(fused[region].mean())
        ig_score = float(ig_smooth[region].mean())
        ada_score = float(adasise[region].mean())
        grad_score = float(gradcam[region].mean())
        score = (
            0.60 * ig_score
            + 0.30 * ada_score
            + 0.10 * grad_score
            + 0.20 * fused_score
        )
        components.append((score, label))

    components = sorted(components, reverse=True)
    keep_labels = [label for _, label in components[:top_k_components]]

    # Critical fallback: the component filter is experimental and must never
    # erase a nonzero fused explanation just because no component survived.
    if len(keep_labels) == 0:
        COMPONENT_FILTER_FALLBACK_COUNT += 1
        return percentile_normalize(fused)

    filtered = np.zeros_like(fused, dtype=np.float32)
    for label in keep_labels:
        filtered[labels == label] = fused[labels == label]

    # Second safety fallback: if filtering somehow empties the map, keep the
    # pre-filter fusion so downstream metrics can honestly evaluate it.
    if float(np.max(filtered)) <= 1e-8 or float(np.sum(filtered)) <= 1e-8:
        COMPONENT_FILTER_FALLBACK_COUNT += 1
        return percentile_normalize(fused)

    return percentile_normalize(filtered)

def calculate_spearman(map1, map2):
    map1 = ensure_2d_float_map(map1)
    map2 = resize_to_match(map2, map1.shape)
    corr, _ = spearmanr(map1.flatten(), map2.flatten())
    return 0.0 if np.isnan(corr) else corr

def prepare_heatmap_and_mask_for_metric(heatmap, mask_binary, eps=1e-8):
    hm = ensure_2d_float_map(heatmap)
    mask = ensure_2d_float_map(mask_binary)

    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0).astype(np.float32)

    if mask.shape != hm.shape:
        mask = cv2.resize(
            mask.astype(np.float32),
            (hm.shape[1], hm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0.5).astype(np.float32)

    if float(np.max(hm)) <= eps:
        return hm, mask, False

    hm = hm / (float(np.max(hm)) + eps)
    return hm, mask, True

def soft_pointing_game(heatmap, mask_binary, eps=1e-8):
    hm, mask, valid = prepare_heatmap_and_mask_for_metric(heatmap, mask_binary, eps=eps)
    if not valid:
        return 0.0
    global_peak = float(np.max(hm))
    in_mask_peak = float(np.max(hm * mask))
    return float(np.clip(in_mask_peak / (global_peak + eps), 0.0, 1.0))

def topk_pointing_game(heatmap, mask_binary, top_percent=1.0, eps=1e-8):
    hm, mask, valid = prepare_heatmap_and_mask_for_metric(heatmap, mask_binary, eps=eps)
    if not valid:
        return 0.0
    flat = hm.reshape(-1)
    mask_flat = mask.reshape(-1)
    n_pixels = flat.size
    k = max(1, int(round((top_percent / 100.0) * n_pixels)))
    top_idx = np.argsort(flat)[::-1][:k]
    return 1.0 if np.any(mask_flat[top_idx] > 0.5) else 0.0

def saliency_precision_at_k(heatmap, mask_binary, top_percent=1.0, eps=1e-8):
    hm, mask, valid = prepare_heatmap_and_mask_for_metric(heatmap, mask_binary, eps=eps)
    if not valid:
        return 0.0
    flat = hm.reshape(-1)
    mask_flat = mask.reshape(-1)
    n_pixels = flat.size
    k = max(1, int(round((top_percent / 100.0) * n_pixels)))
    top_idx = np.argsort(flat)[::-1][:k]
    return float(np.mean(mask_flat[top_idx] > 0.5))

def distance_pointing_game(heatmap, mask_binary, sigma=20.0, eps=1e-8):
    hm, mask, valid = prepare_heatmap_and_mask_for_metric(heatmap, mask_binary, eps=eps)
    if not valid or float(np.sum(mask)) <= eps:
        return 0.0
    max_y, max_x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    inverse_mask = 1 - mask_uint8
    distance_map = cv2.distanceTransform(inverse_mask, cv2.DIST_L2, 5)
    distance = float(distance_map[max_y, max_x])
    score = np.exp(-distance / sigma)
    return float(np.clip(score, 0.0, 1.0))

def softmax_dict(score_dict, temperature=0.25, eps=1e-8):
    names = list(score_dict.keys())
    scores = np.asarray([score_dict[n] for n in names], dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    temperature = max(float(temperature), eps)
    scores = scores / temperature
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    denom = np.sum(exp_scores)
    if denom <= eps:
        return {name: 1.0 / len(names) for name in names}
    probs = exp_scores / denom
    return {name: float(prob) for name, prob in zip(names, probs)}

def compactness_score(heatmap, eps=1e-8):
    hm = percentile_normalize(heatmap)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)
    if float(np.max(hm)) <= eps:
        return 0.0
    flat = hm.reshape(-1)
    n = flat.size
    l1 = float(np.sum(np.abs(flat)))
    l2 = float(np.sqrt(np.sum(flat ** 2)))
    if l2 <= eps or n <= 1:
        return 0.0
    sqrt_n = float(np.sqrt(n))
    hoyer = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0)
    hoyer = float(np.clip(hoyer, 0.0, 1.0))
    active_ratio = float(np.mean(hm > 0.5))
    if active_ratio < 0.0005:
        hoyer *= 0.75
    return hoyer

def peak_dominance_score(heatmap, eps=1e-8):
    hm = percentile_normalize(heatmap)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)
    if float(np.max(hm)) <= eps:
        return 0.0
    high = np.percentile(hm, 99)
    median = np.median(hm)
    score = (high - median) / (high + eps)
    return float(np.clip(score, 0.0, 1.0))

def retina_containment_score(heatmap, retina_mask=None, eps=1e-8):
    hm = percentile_normalize(heatmap)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)
    if float(np.sum(hm)) <= eps:
        return 0.0
    if retina_mask is None:
        return 1.0
    mask = resize_to_match(retina_mask, hm.shape, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    return float(np.sum(hm * mask) / (np.sum(hm) + eps))

def agreement_score(name, normalized_maps):
    target = normalized_maps[name]
    others = [v for k, v in normalized_maps.items() if k != name]
    if len(others) == 0:
        return 0.5
    scores = []
    for other in others:
        other = resize_to_match(other, target.shape)
        corr = calculate_spearman(target, other)
        corr = (corr + 1.0) / 2.0
        scores.append(float(np.clip(corr, 0.0, 1.0)))
    return float(np.mean(scores))

def quick_deletion_reliability(model, image_tensor, heatmap, remove_percent=5.0, eps=1e-8):
    if model is None or image_tensor is None:
        return 0.5
    model.eval()
    hm = ensure_2d_float_map(heatmap)
    _, _, h, w = image_tensor.shape
    if hm.shape != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.maximum(hm, 0.0)
    if float(np.max(hm)) <= eps:
        return 0.0
    hm = hm / (float(np.max(hm)) + eps)
    flat = hm.reshape(-1)
    n_pixels = flat.size
    k = max(1, int(round((remove_percent / 100.0) * n_pixels)))
    top_idx = np.argsort(flat)[::-1][:k]
    selected = np.zeros(n_pixels, dtype=np.float32)
    selected[top_idx] = 1.0
    mask_2d = selected.reshape(h, w)
    mask_tensor = torch.from_numpy(mask_2d).to(device=image_tensor.device, dtype=image_tensor.dtype).unsqueeze(0).unsqueeze(0)
    baseline_tensor = torch.zeros_like(image_tensor)
    def raw_score(x):
        out = model(x)
        return float(out.view(-1)[0].detach().cpu().item())
    with torch.no_grad():
        original_score = raw_score(image_tensor)
        baseline_score = raw_score(baseline_tensor)
        deleted = image_tensor * (1.0 - mask_tensor) + baseline_tensor * mask_tensor
        deleted_score = raw_score(deleted)
    denom = abs(original_score - baseline_score)
    if denom <= eps:
        return 0.5
    drop = abs(original_score - deleted_score) / (denom + eps)
    return float(np.clip(drop, 0.0, 1.0))

def soft_top_percentile_filter(heatmap, keep_percentile=60.0, eps=1e-8):
    hm = percentile_normalize(heatmap)
    if float(np.max(hm)) <= eps:
        return hm
    threshold = np.percentile(hm, keep_percentile)
    hm = np.maximum(hm - threshold, 0.0)
    if float(np.max(hm)) <= eps:
        return percentile_normalize(heatmap)
    return percentile_normalize(hm)

# ==========================================
# CCEM FUSION ALGORITHM
# ==========================================
def legacy_soft_union_ccem(
    G,
    AdaS,
    IG,
    retina_mask=None,
    tau=0.50,
    weights=(0.30, 0.40, 0.30),
    blur_sigma=1.0,
    apply_component_filter=True,
):
    """
    Legacy soft-union CCEM. Kept for reproducibility; new experiments should use
    ig_anchored_ccem or generate_ccem(..., mode="ig_anchored").
    """
    cam_weight, adasise_weight, smoothig_weight = weights
    reference_shape = retina_mask.shape if retina_mask is not None else ensure_2d_float_map(G).shape

    cam = np.clip(resize_like(normalize_heatmap(G), reference_shape), 0.0, 1.0)
    adasise = np.clip(resize_like(normalize_heatmap(AdaS), reference_shape), 0.0, 1.0)
    smoothig = np.clip(resize_like(normalize_heatmap(IG), reference_shape), 0.0, 1.0)

    soft_union = 1.0 - (
        ((1.0 - cam) ** cam_weight)
        * ((1.0 - adasise) ** adasise_weight)
        * ((1.0 - smoothig) ** smoothig_weight)
    )

    agreement = (
        (cam > tau).astype(np.float32)
        + (adasise > tau).astype(np.float32)
        + (smoothig > tau).astype(np.float32)
    ) / 3.0

    fused = soft_union * (0.75 + 0.25 * agreement)

    if retina_mask is not None:
        mask = resize_like(np.asarray(retina_mask, dtype=np.float32), fused.shape)
        mask = (mask > 0).astype(np.float32)
        fused = fused * mask
    else:
        mask = None

    if apply_component_filter:
        fused = connected_component_filter(fused)

    if blur_sigma is not None and blur_sigma > 0:
        fused = cv2.GaussianBlur(fused, (0, 0), sigmaX=blur_sigma)

    if mask is not None:
        fused = fused * mask

    return normalize_heatmap(fused)

def ig_anchored_ccem(
    gradcam,
    adasise,
    ig_smooth,
    image=None,
    retina_mask=None,
    w_gradcam=0.05,
    w_adasise=0.15,
    w_ig=0.80,
    ig_gate_threshold=0.15,
    agreement_threshold=0.80,
    gate_floor=0.80,
    agreement_bonus=0.05,
    blur_sigma=1.0,
    apply_component_filter=False,
    threshold_percentile=88,
    min_area=20,
    max_area_ratio=0.10,
    top_k_components=8,
):
    target_shape = ensure_2d_float_map(ig_smooth).shape[:2]

    gradcam = resize_to_match(gradcam, target_shape)
    adasise = resize_to_match(adasise, target_shape)
    ig_smooth = resize_to_match(ig_smooth, target_shape)

    # Use percentile normalization, not rank normalization, as the default.
    # Rank normalization can destroy magnitude information and over-flatten maps.
    g = percentile_normalize(gradcam)
    a = percentile_normalize(adasise)
    i = percentile_normalize(ig_smooth)

    weight_sum = w_gradcam + w_adasise + w_ig
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")

    w_gradcam = w_gradcam / weight_sum
    w_adasise = w_adasise / weight_sum
    w_ig = w_ig / weight_sum

    # Preserve IG as the main explanation signal.
    fused = w_gradcam * g + w_adasise * a + w_ig * i

    # Soft IG gate. This should modulate the fusion, not erase it.
    ig_gate = np.clip(i / (ig_gate_threshold + 1e-8), 0.0, 1.0)
    fused = fused * (gate_floor + (1.0 - gate_floor) * ig_gate)

    # Mild agreement bonus only.
    agreement = (
        (g > agreement_threshold).astype(np.float32)
        + (a > agreement_threshold).astype(np.float32)
        + (i > agreement_threshold).astype(np.float32)
    ) / 3.0
    fused = fused * ((1.0 - agreement_bonus) + agreement_bonus * agreement)

    if retina_mask is None and image is not None:
        retina_mask = make_retina_mask(image)

    if retina_mask is not None:
        retina_mask = resize_to_match(retina_mask, target_shape)
        retina_mask = (retina_mask > 0.5).astype(np.float32)
        fused = fused * retina_mask

    if blur_sigma is not None and blur_sigma > 0:
        fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), sigmaX=blur_sigma)

    fused = percentile_normalize(fused)

    if apply_component_filter:
        pre_filter = fused.copy()
        fused = component_filter_fusion(
            fused=fused,
            gradcam=g,
            adasise=a,
            ig_smooth=i,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            top_k_components=top_k_components,
            threshold_percentile=threshold_percentile,
        )

        if float(np.max(fused)) <= 1e-8 or float(np.sum(fused)) <= 1e-8:
            fused = pre_filter

    # Final safety fallback: if the fused map is somehow empty but IG is not,
    # return IG rather than an empty explanation.
    if float(np.max(fused)) <= 1e-8 or float(np.sum(fused)) <= 1e-8:
        if float(np.max(i)) > 1e-8 and float(np.sum(i)) > 1e-8:
            fused = i

    return percentile_normalize(fused)

def adaptive_reliability_ccem(
    gradcam,
    adasise,
    ig_smooth,
    image=None,
    image_tensor=None,
    model=None,
    retina_mask=None,
    temperature=0.25,
    faithfulness_weight=0.35,
    agreement_weight=0.20,
    compactness_weight=0.20,
    containment_weight=0.15,
    peak_weight=0.10,
    sharpen_gamma=1.5,
    soft_keep_percentile=None,
    blur_sigma=0.8,
    return_debug=False,
):
    target_shape = ensure_2d_float_map(ig_smooth).shape[:2]

    raw_maps = {
        "GradCAM++": resize_to_match(gradcam, target_shape),
        "Ada-SISE": resize_to_match(adasise, target_shape),
        "IG_Smooth": resize_to_match(ig_smooth, target_shape),
    }

    normalized_maps = {name: robust_normalize(hm) for name, hm in raw_maps.items()}

    if retina_mask is None and image is not None:
        retina_mask = make_retina_mask(image)

    if retina_mask is not None:
        retina_mask = resize_to_match(retina_mask, target_shape, interpolation=cv2.INTER_NEAREST)
        retina_mask = (retina_mask > 0.5).astype(np.float32)

    reliability = {}

    for name, hm in normalized_maps.items():
        faith = quick_deletion_reliability(
            model=model,
            image_tensor=image_tensor,
            heatmap=hm,
            remove_percent=5.0,
        )
        agree = agreement_score(name=name, normalized_maps=normalized_maps)
        compact = compactness_score(hm)
        contain = retina_containment_score(heatmap=hm, retina_mask=retina_mask)
        peak = peak_dominance_score(hm)

        score = (
            faithfulness_weight * faith
            + agreement_weight * agree
            + compactness_weight * compact
            + containment_weight * contain
            + peak_weight * peak
        )

        reliability[name] = float(score)

    weights = softmax_dict(reliability, temperature=temperature)

    fused = np.zeros(target_shape, dtype=np.float32)
    for name, hm in normalized_maps.items():
        fused += weights[name] * hm

    if retina_mask is not None:
        fused = fused * retina_mask

    if blur_sigma is not None and blur_sigma > 0:
        fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), sigmaX=blur_sigma)

    fused = robust_normalize(fused)

    if sharpen_gamma is not None and sharpen_gamma > 1.0:
        fused = np.power(fused, sharpen_gamma)
        fused = robust_normalize(fused)

    if soft_keep_percentile is not None:
        fused = soft_top_percentile_filter(fused, keep_percentile=soft_keep_percentile)

    if float(np.max(fused)) <= 1e-8 or float(np.sum(fused)) <= 1e-8:
        # Fusion collapsed (e.g. blur/sharpen/keep-percentile emptied a sparse
        # map). Fall back to the most reliable source map, re-normalized with
        # robust_normalize so a sparse-but-real map isn't discarded again.
        for name in sorted(reliability, key=reliability.get, reverse=True):
            candidate = robust_normalize(raw_maps[name])
            if retina_mask is not None:
                masked = robust_normalize(candidate * retina_mask)
                if float(np.max(masked)) > 1e-8:
                    candidate = masked
            if float(np.max(candidate)) > 1e-8:
                fused = candidate
                break

    if return_debug:
        return fused, {"weights": weights, "reliability": reliability}

    return fused

def generate_ccem(
    G,
    AdaS,
    IG,
    retina_mask=None,
    image=None,
    model=None,
    image_tensor=None,
    mode="ig_anchored",
    tau=0.50,
    weights=(0.05, 0.15, 0.80),
    blur_sigma=1.0,
    apply_component_filter=False,
    ig_gate_threshold=0.15,
    agreement_threshold=0.80,
    gate_floor=0.80,
    agreement_bonus=0.05,
    threshold_percentile=88,
    min_area=20,
    max_area_ratio=0.10,
    top_k_components=8,
    adaptive_temperature=0.25,
    adaptive_faithfulness_weight=0.35,
    adaptive_agreement_weight=0.20,
    adaptive_compactness_weight=0.20,
    adaptive_containment_weight=0.15,
    adaptive_peak_weight=0.10,
    adaptive_sharpen_gamma=1.5,
    adaptive_soft_keep_percentile=None,
    return_debug=False,
):
    """
    Public CCEM entry point. Defaults to IG-anchored fusion; pass
    mode="legacy_soft_union" to reproduce the previous Noisy-OR behavior.
    """
    if mode == "ig_only":
        return percentile_normalize(IG)

    if mode in ("simple_average", "soft_agreement"):
        target_shape = ensure_2d_float_map(IG).shape[:2]

        g = percentile_normalize(resize_to_match(G, target_shape))
        a = percentile_normalize(resize_to_match(AdaS, target_shape))
        i = percentile_normalize(resize_to_match(IG, target_shape))

        w_gradcam, w_adasise, w_ig = weights
        weight_sum = w_gradcam + w_adasise + w_ig
        if weight_sum <= 0:
            raise ValueError("Fusion weights must sum to a positive value.")

        w_gradcam /= weight_sum
        w_adasise /= weight_sum
        w_ig /= weight_sum

        fused = w_gradcam * g + w_adasise * a + w_ig * i

        if mode == "soft_agreement":
            agreement = (
                (g > agreement_threshold).astype(np.float32)
                + (a > agreement_threshold).astype(np.float32)
                + (i > agreement_threshold).astype(np.float32)
            ) / 3.0
            fused = fused * ((1.0 - agreement_bonus) + agreement_bonus * agreement)

        if retina_mask is not None:
            mask = resize_to_match(retina_mask, target_shape)
            mask = (mask > 0.5).astype(np.float32)
            fused = fused * mask

        return percentile_normalize(fused)

    if mode in ("adaptive", "adaptive_reliability", "adaptive_ccem"):
        return adaptive_reliability_ccem(
            gradcam=G,
            adasise=AdaS,
            ig_smooth=IG,
            image=image,
            image_tensor=image_tensor,
            model=model,
            retina_mask=retina_mask,
            temperature=adaptive_temperature,
            faithfulness_weight=adaptive_faithfulness_weight,
            agreement_weight=adaptive_agreement_weight,
            compactness_weight=adaptive_compactness_weight,
            containment_weight=adaptive_containment_weight,
            peak_weight=adaptive_peak_weight,
            sharpen_gamma=adaptive_sharpen_gamma,
            soft_keep_percentile=adaptive_soft_keep_percentile,
            blur_sigma=blur_sigma,
            return_debug=return_debug,
        )

    if mode in ("legacy", "legacy_soft_union", "soft_union", "noisy_or"):
        return legacy_soft_union_ccem(
            G,
            AdaS,
            IG,
            retina_mask=retina_mask,
            tau=tau,
            weights=weights,
            blur_sigma=blur_sigma,
            apply_component_filter=apply_component_filter,
        )

    if mode not in ("ig_anchored", "ig", "anchored"):
        raise ValueError(f"Unknown CCEM mode: {mode}")

    w_gradcam, w_adasise, w_ig = weights
    return ig_anchored_ccem(
        gradcam=G,
        adasise=AdaS,
        ig_smooth=IG,
        image=image,
        retina_mask=retina_mask,
        w_gradcam=w_gradcam,
        w_adasise=w_adasise,
        w_ig=w_ig,
        ig_gate_threshold=ig_gate_threshold,
        agreement_threshold=agreement_threshold,
        gate_floor=gate_floor,
        agreement_bonus=agreement_bonus,
        blur_sigma=blur_sigma,
        apply_component_filter=apply_component_filter,
        threshold_percentile=threshold_percentile,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        top_k_components=top_k_components,
    )

def heatmap_for_overlay(heatmap, lower_percentile=2.0, upper_percentile=99.0):
    return normalize_heatmap(
        heatmap,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

# ==========================================
# ODExAI EVALUATION METRICS
# ==========================================
def calculate_advanced_metrics(
    model,
    image_tensor,
    heatmap,
    mask_binary,
    n_steps=20,
    eps=1e-8,
):
    """
    Corrected XAI metrics for non-negative 2D saliency maps.

    Metrics:
    - EBPG: fraction of saliency energy inside expert lesion mask.
    - PG: whether the max-saliency pixel falls inside expert lesion mask.
    - Sparsity: Hoyer sparsity, bounded in [0, 1].
    - Deletion AUC: preservation of original model score as salient pixels are removed.
    - Insertion AUC: recovery of original model score as salient pixels are inserted.
    - OA: insertion AUC - deletion AUC.

    Assumptions:
    - image_tensor shape: [1, C, H, W]
    - heatmap shape: [H, W], or resizable to [H, W]
    - mask_binary shape: [H, W], or resizable to [H, W]
    """
    model.eval()

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            f"image_tensor must have shape [1, C, H, W], got {tuple(image_tensor.shape)}"
        )

    _, _, height, width = image_tensor.shape

    heatmap = ensure_2d_float_map(heatmap)
    mask_binary = ensure_2d_float_map(mask_binary)

    if heatmap.shape != (height, width):
        heatmap = cv2.resize(
            heatmap,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    if mask_binary.shape != (height, width):
        mask_binary = cv2.resize(
            mask_binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0).astype(np.float32)
    mask_binary = np.nan_to_num(mask_binary, nan=0.0, posinf=0.0, neginf=0.0)
    mask_binary = (mask_binary > 0.5).astype(np.float32)

    hm_max = float(np.max(heatmap)) if heatmap.size else 0.0
    if hm_max <= eps:
        return 0.0, 0.0, 0.0, np.nan, np.nan, np.nan

    heatmap = heatmap / (hm_max + eps)
    flat_heatmap = heatmap.reshape(-1)
    n_pixels = flat_heatmap.size

    # 1. LOCALIZATION: EBPG (Energy-Based Pointing Game)
    saliency_sum = float(np.sum(heatmap))
    ebpg = float(np.sum(heatmap * mask_binary) / (saliency_sum + eps))

    # 2. LOCALIZATION: PG (Pointing Game)
    max_idx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    pg = 1.0 if mask_binary[max_idx] > 0.0 else 0.0

    # 3. COMPLEXITY: Sparsity (Hoyer measure)
    l1_norm = float(np.sum(np.abs(flat_heatmap)))
    l2_norm = float(np.sqrt(np.sum(flat_heatmap ** 2)))
    if l2_norm <= eps or n_pixels <= 1:
        sparsity = 0.0
    else:
        sqrt_n = float(np.sqrt(n_pixels))
        sparsity = (sqrt_n - (l1_norm / l2_norm)) / (sqrt_n - 1.0)
        sparsity = float(np.clip(sparsity, 0.0, 1.0))

    # 4. FAITHFULNESS: Deletion & Insertion
    sorted_indices = np.argsort(flat_heatmap)[::-1]
    n_steps = max(int(n_steps), 1)
    fractions = np.linspace(0.0, 1.0, n_steps + 1)

    del_scores = []
    ins_scores = []
    baseline_tensor = torch.zeros_like(image_tensor)

    def raw_model_score(x):
        out = model(x)
        return float(out.view(-1)[0].detach().cpu().item())

    with torch.no_grad():
        original_score = raw_model_score(image_tensor)
        baseline_score = raw_model_score(baseline_tensor)

    denom = abs(original_score - baseline_score)
    if denom <= eps:
        return ebpg, pg, sparsity, np.nan, np.nan, np.nan

    direction = 1.0 if original_score >= baseline_score else -1.0

    def normalized_preservation_score(x):
        score = raw_model_score(x)
        normalized = direction * (score - baseline_score) / denom
        return float(np.clip(normalized, 0.0, 1.0))

    with torch.no_grad():
        for fraction in fractions:
            num_pixels = int(round(float(fraction) * n_pixels))
            selected = np.zeros(n_pixels, dtype=np.float32)
            if num_pixels > 0:
                selected[sorted_indices[:num_pixels]] = 1.0

            mask_2d = selected.reshape(height, width)
            mask_tensor = torch.from_numpy(mask_2d).to(
                device=image_tensor.device,
                dtype=image_tensor.dtype,
            ).unsqueeze(0).unsqueeze(0)

            deleted_image = image_tensor * (1.0 - mask_tensor) + baseline_tensor * mask_tensor
            inserted_image = baseline_tensor * (1.0 - mask_tensor) + image_tensor * mask_tensor

            del_scores.append(normalized_preservation_score(deleted_image))
            ins_scores.append(normalized_preservation_score(inserted_image))

    if hasattr(np, "trapezoid"):
        del_auc = float(np.trapezoid(del_scores, fractions))
        ins_auc = float(np.trapezoid(ins_scores, fractions))
    else:
        del_auc = float(np.trapz(del_scores, fractions))
        ins_auc = float(np.trapz(ins_scores, fractions))

    # 5. FAITHFULNESS: Over-All (OA)
    oa = float(ins_auc - del_auc)

    return ebpg, pg, sparsity, del_auc, ins_auc, oa

def calculate_extended_metrics(
    model,
    image_tensor,
    heatmap,
    mask_binary,
    n_steps=20,
    eps=1e-8,
):
    ebpg, pg, sparsity, del_auc, ins_auc, oa = calculate_advanced_metrics(
        model=model,
        image_tensor=image_tensor,
        heatmap=heatmap,
        mask_binary=mask_binary,
        n_steps=n_steps,
        eps=eps,
    )

    soft_pg = soft_pointing_game(heatmap=heatmap, mask_binary=mask_binary, eps=eps)
    topk_pg_05 = topk_pointing_game(heatmap=heatmap, mask_binary=mask_binary, top_percent=0.5, eps=eps)
    topk_pg_1 = topk_pointing_game(heatmap=heatmap, mask_binary=mask_binary, top_percent=1.0, eps=eps)
    topk_pg_2 = topk_pointing_game(heatmap=heatmap, mask_binary=mask_binary, top_percent=2.0, eps=eps)
    precision_05 = saliency_precision_at_k(heatmap=heatmap, mask_binary=mask_binary, top_percent=0.5, eps=eps)
    precision_1 = saliency_precision_at_k(heatmap=heatmap, mask_binary=mask_binary, top_percent=1.0, eps=eps)
    precision_2 = saliency_precision_at_k(heatmap=heatmap, mask_binary=mask_binary, top_percent=2.0, eps=eps)
    distance_pg = distance_pointing_game(heatmap=heatmap, mask_binary=mask_binary, sigma=20.0, eps=eps)

    return {
        "EBPG": ebpg,
        "PG": pg,
        "SoftPG": soft_pg,
        "TopKPG_0_5": topk_pg_05,
        "TopKPG_1": topk_pg_1,
        "TopKPG_2": topk_pg_2,
        "Precision_0_5": precision_05,
        "Precision_1": precision_1,
        "Precision_2": precision_2,
        "DistancePG": distance_pg,
        "Sparsity": sparsity,
        "Deletion": del_auc,
        "Insertion": ins_auc,
        "OA": oa,
    }
