
import cv2
import math
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.filters import gabor, prewitt
from skimage.measure import regionprops, label

def clean_value(x):
    try:
        if np.isnan(x) or np.isinf(x):
            return 0.0
    except Exception:
        return 0.0
    return float(x)

def safe_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        clean_value(np.mean(values)),
        clean_value(np.std(values)),
        clean_value(skew(values, nan_policy="omit")),
        clean_value(kurtosis(values, nan_policy="omit")),
    )

def get_nonblack_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 5).astype(np.uint8)
    if np.sum(mask) == 0:
        mask = np.ones(gray.shape, dtype=np.uint8)
    return mask

def crop_nonblack_region(rgb, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgb
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return rgb[y1:y2 + 1, x1:x2 + 1].copy()

def extract_glcm(gray, mask):
    feats = {}
    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0
    gray_q = (gray_masked / 8).astype(np.uint8)

    distances = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    props = ["contrast", "correlation", "energy", "homogeneity", "ASM", "dissimilarity"]

    glcm = graycomatrix(gray_q, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    for prop in props:
        values = graycoprops(glcm, prop).flatten()
        for i, v in enumerate(values):
            feats[f"glcm_{prop}_{i}"] = clean_value(v)

    return feats

def extract_lbp(gray, mask):
    feats = {}
    for P, R in [(8, 1), (16, 2), (24, 3)]:
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        values = lbp[mask > 0]
        hist, _ = np.histogram(values, bins=10, range=(0, P + 2), density=True)
        for i, v in enumerate(hist):
            feats[f"lbp_P{P}_R{R}_{i}"] = clean_value(v)
    return feats

def extract_gabor(gray, mask):
    feats = {}
    gray_f = gray.astype(np.float32) / 255.0
    idx = 0
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for sigma in [1, 3]:
            for freq in [0.1, 0.3]:
                real, imag = gabor(gray_f, frequency=freq, theta=theta, sigma_x=sigma, sigma_y=sigma)
                mag = np.sqrt(real ** 2 + imag ** 2)
                feats[f"gabor_mean_{idx}"] = clean_value(np.mean(mag[mask > 0]))
                idx += 1
    return feats

def extract_hog_features(gray, mask):
    feats = {}
    rgb_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    crop = crop_nonblack_region(rgb_gray, mask)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.resize(crop_gray, (128, 128))

    h = hog(
        crop_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    for i, chunk in enumerate(np.array_split(h, 10)):
        feats[f"hog_{i}"] = clean_value(np.mean(chunk))

    return feats

def extract_laws(gray, mask):
    feats = {}
    vectors = {
        "L5": np.array([1, 4, 6, 4, 1]),
        "E5": np.array([-1, -2, 0, 2, 1]),
        "S5": np.array([-1, 0, 2, 0, -1]),
        "R5": np.array([1, -4, 6, -4, 1]),
    }

    gray_f = gray.astype(np.float32)

    for n1, v1 in vectors.items():
        for n2, v2 in vectors.items():
            kernel = np.outer(v1, v2)
            response = cv2.filter2D(gray_f, -1, kernel)
            energy = np.abs(response)
            feats[f"laws_{n1}_{n2}"] = clean_value(np.mean(energy[mask > 0]))

    return feats

def extract_color(rgb, mask):
    feats = {}
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    rgb_pixels = rgb[mask > 0]
    hsv_pixels = hsv[mask > 0]

    rgb_hist, _ = np.histogramdd(
        rgb_pixels,
        bins=(4, 4, 4),
        range=((0, 256), (0, 256), (0, 256)),
        density=True
    )

    hsv_hist, _ = np.histogramdd(
        hsv_pixels,
        bins=(4, 4, 4),
        range=((0, 180), (0, 256), (0, 256)),
        density=True
    )

    for i, v in enumerate(rgb_hist.flatten()):
        feats[f"rgb_hist_{i}"] = clean_value(v)

    for i, v in enumerate(hsv_hist.flatten()):
        feats[f"hsv_hist_{i}"] = clean_value(v)

    for name, pixels in [("rgb", rgb_pixels), ("hsv", hsv_pixels)]:
        for ch in range(3):
            mean, std, sk, ku = safe_stats(pixels[:, ch])
            feats[f"{name}_ch{ch}_mean"] = mean
            feats[f"{name}_ch{ch}_std"] = std
            feats[f"{name}_ch{ch}_skew"] = sk
            feats[f"{name}_ch{ch}_kurtosis"] = ku

    if len(rgb_pixels) > 2:
        corr = np.corrcoef(rgb_pixels.T)
        feats["corr_rg"] = clean_value(corr[0, 1])
        feats["corr_rb"] = clean_value(corr[0, 2])
        feats["corr_gb"] = clean_value(corr[1, 2])
    else:
        feats["corr_rg"] = 0.0
        feats["corr_rb"] = 0.0
        feats["corr_gb"] = 0.0

    return feats

def extract_shape(mask):
    feats = {}
    binary = (mask > 0).astype(np.uint8)
    area = np.sum(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0 or area == 0:
        for k in ["area", "perimeter", "circularity", "eccentricity", "solidity", "extent", "compactness", "aspect_ratio"]:
            feats[f"shape_{k}"] = 0.0
        return feats

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0.0
    circularity = (4 * math.pi * area) / (perimeter ** 2 + 1e-8)
    compactness = (perimeter ** 2) / (area + 1e-8)

    props = regionprops(label(binary))

    if len(props) > 0:
        prop = max(props, key=lambda p: p.area)
        eccentricity = prop.eccentricity
        solidity = prop.solidity
        extent = prop.extent
    else:
        eccentricity = 0.0
        solidity = 0.0
        extent = 0.0

    feats["shape_area"] = clean_value(area)
    feats["shape_perimeter"] = clean_value(perimeter)
    feats["shape_circularity"] = clean_value(circularity)
    feats["shape_eccentricity"] = clean_value(eccentricity)
    feats["shape_solidity"] = clean_value(solidity)
    feats["shape_extent"] = clean_value(extent)
    feats["shape_compactness"] = clean_value(compactness)
    feats["shape_aspect_ratio"] = clean_value(aspect_ratio)

    return feats

def extract_edges(gray, mask):
    feats = {}
    lesion_area = np.sum(mask > 0)

    canny = cv2.Canny(gray, 100, 200)
    feats["edge_canny_density"] = clean_value(np.sum(canny[mask > 0] > 0) / (lesion_area + 1e-8))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    feats["edge_sobel_mean"] = clean_value(np.mean(sobel_mag[mask > 0]))
    feats["edge_sobel_std"] = clean_value(np.std(sobel_mag[mask > 0]))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    feats["edge_laplacian_mean"] = clean_value(np.mean(np.abs(lap[mask > 0])))

    pre = prewitt(gray.astype(np.float32) / 255.0)
    feats["edge_prewitt_mean"] = clean_value(np.mean(np.abs(pre[mask > 0])))

    return feats

def extract_all_handcrafted_features(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = get_nonblack_mask(rgb)

    features = {}
    features.update(extract_glcm(gray, mask))
    features.update(extract_lbp(gray, mask))
    features.update(extract_gabor(gray, mask))
    features.update(extract_hog_features(gray, mask))
    features.update(extract_laws(gray, mask))
    features.update(extract_color(rgb, mask))
    features.update(extract_shape(mask))
    features.update(extract_edges(gray, mask))

    return features
