"""
Automated OCR Document Scanner
--------------------------------
Implements the 4-phase pipeline described in the project synopsis:
 
  Phase 1 - Preprocessing & Morphology: grayscale, Gaussian blur,
            Canny edge detection, morphological dilation.
  Phase 2 - Geometric Isolation: contour extraction (border following),
            approxPolyDP for the 4-point document quadrilateral, with
            a minAreaRect fallback for jagged/damaged edges.
  Phase 3 - Perspective Transformation: corner ordering + homography
            (getPerspectiveTransform) to warp to a flat top-down view.
  Phase 4 - Binarization & OCR: compares Adaptive Gaussian Thresholding,
            Otsu's global method, and Sauvola's local method, then runs
            Tesseract OCR on each.
 
Also implements the Evaluation Plan from the synopsis: Character Error
Rate (CER) and Word Error Rate (WER) against hand-typed ground truth,
for the raw photograph vs. each processed/binarized variant.
 
Usage:
  Single image:
    python3 scanner_ocr.py <image_path> [output_dir]
 
  Batch (a whole folder of .jpg/.jpeg/.png files):
    python3 scanner_ocr.py <input_dir> [output_dir]
 
  With evaluation (CER/WER against ground truth .txt files, one per
  image, same base filename -- e.g. receipt1.jpg -> receipt1.txt):
    python3 scanner_ocr.py <input_path> [output_dir] --ground-truth <gt_dir>
"""
 
import cv2
import numpy as np
import pytesseract
import sys
import os
import csv
import argparse
import glob
 
try:
    from skimage.filters import threshold_sauvola
    SAUVOLA_AVAILABLE = True
except ImportError:
    SAUVOLA_AVAILABLE = False
 
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
 
# Tesseract page segmentation mode. 6 = one uniform block of text, which
# keeps multi-column receipts/invoices (item + price) together. The
# default mode 3 can silently drop a separate column such as prices.
OCR_PSM = 6
 
 
# =========================================================
# Phase 2/3 helpers: corner ordering + perspective transform
# =========================================================
 
def order_points(pts):
    """Sorts 4 corners into: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left     -> smallest x+y
    rect[2] = pts[np.argmax(s)]        # bottom-right -> largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # top-right    -> smallest y-x
    rect[3] = pts[np.argmax(diff)]     # bottom-left  -> largest y-x
    return rect
 
 
def warp_perspective(image, pts):
    """Warps the angled document into a flat, top-down view (homography)."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
 
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    return warped
 
 
# =========================================================
# Phase 1/2: document boundary detection
# =========================================================
 
def _largest_quad(contours, frame_area, min_frac=0.15):
    """Return the largest contour that approximates to a clean 4-point
    quadrilateral covering at least `min_frac` of the frame, else None."""
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_frac * frame_area:
            return approx.reshape(4, 2).astype("float32")
    return None
 
 
def find_document_contour(gray):
    """
    Locate the document's 4 corners by trying several strategies, in
    order, and using the first one that yields a clean 4-point quad
    covering a plausible fraction of the frame:
 
      1. Low-threshold Canny: catches faint edges (e.g. a white page
         on a light-gray background where the two tones are only
         ~20 levels apart) that a default/high Canny threshold misses.
      2. Standard Canny: catches normal higher-contrast page edges
         (dark desk/table background, strong shadows, etc).
      3. Brightness threshold (Otsu): works when the page is a
         uniformly bright blob well-separated from the background
         tone.
 
    Falls back to minAreaRect (rotated bounding box) of the largest
    blob/contour found by whichever strategy got furthest, if no
    clean 4-point quad is found by any of the above. This matches
    Phase 2 of the synopsis (approxPolyDP with minAreaRect fallback).
    """
    frame_area = gray.shape[0] * gray.shape[1]
    blurred5 = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred7 = cv2.GaussianBlur(gray, (7, 7), 0)
 
    all_contours = []
 
    # --- Strategy 1: low-threshold Canny (subtle page/background contrast) ---
    edges_low = cv2.Canny(blurred5, 10, 30)
    edges_low = cv2.dilate(edges_low, np.ones((5, 5), np.uint8), iterations=2)
    c_low, _ = cv2.findContours(edges_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours += list(c_low)
    quad = _largest_quad(c_low, frame_area)
    if quad is not None:
        return quad, "low-threshold Canny edges"
 
    # --- Strategy 2: standard Canny (normal contrast) ---
    edges_std = cv2.Canny(blurred5, 50, 150)
    edges_std = cv2.dilate(edges_std, np.ones((3, 3), np.uint8), iterations=1)
    c_std, _ = cv2.findContours(edges_std, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours += list(c_std)
    quad = _largest_quad(c_std, frame_area)
    if quad is not None:
        return quad, "standard Canny edges"
 
    # --- Strategy 3: Otsu brightness threshold ---
    _, thresh = cv2.threshold(blurred7, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    c_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours += list(c_thresh)
    quad = _largest_quad(c_thresh, frame_area)
    if quad is not None:
        return quad, "brightness threshold (Otsu)"
 
    # --- Fallback: rotated bounding box of the biggest blob seen so far ---
    if all_contours:
        largest = max(all_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype("float32"), "rotated bounding box (minAreaRect fallback)"
 
    return None, None
 
 
# =========================================================
# Phase 4: binarization variants (Adaptive Gaussian / Otsu / Sauvola)
# =========================================================
 
def correct_illumination(gray):
    """
    Removes shadow / flash-glare gradients before binarization.
 
    Standard "shading correction" trick: estimate the smooth background
    illumination by heavily blurring the image (this washes out the thin
    text strokes but keeps the broad shadow/glare gradient), then divide
    the original by that estimate and rescale to 0-255. Flat lighting
    stays ~unchanged; a shadowed corner and a glare hotspot both get
    pulled back toward the same middle gray, which is what lets a small,
    fixed-size adaptive threshold behave consistently across the whole
    page instead of misreading gradient noise as text.
    """
    blur_ksize = max(25, (min(gray.shape[:2]) // 8) | 1)  # odd, scales with image size
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    background = np.where(background == 0, 1, background).astype(np.float32)
    corrected = (gray.astype(np.float32) / background) * 255.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected
 
 
def _adaptive_block_size(gray):
    """Scale the adaptive-threshold window with image size instead of a
    fixed pixel count, so it behaves the same on a 600px and a 6000px
    flattened scan. Must be odd and >= 3."""
    size = max(15, (min(gray.shape[:2]) // 25) | 1)
    return size
 
 
def binarize_variants(flat_gray):
    """
    Produces the binarization variants named in the synopsis so their
    OCR output can be compared. Returns an ordered dict of
    {method_name: binary_uint8_image}.
 
    Each method now runs on an illumination-corrected copy of the image
    (see correct_illumination) so a shadow on one side and a flash glare
    on the other don't each drag a global/local threshold in opposite
    directions. A mild denoise pass runs first so that stray camera
    sensor noise in flat, low-contrast regions doesn't get misread as
    text texture by the local threshold methods.
    """
    variants = {}
 
    denoised = cv2.fastNlMeansDenoising(flat_gray, h=10, templateWindowSize=7, searchWindowSize=21)
    corrected = correct_illumination(denoised)
    block_size = _adaptive_block_size(flat_gray)
 
    variants["adaptive_gaussian"] = cv2.adaptiveThreshold(
        corrected, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=block_size, C=10
    )
 
    _, otsu = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["otsu"] = otsu
 
    if SAUVOLA_AVAILABLE:
        window_size = block_size
        sauvola_thresh = threshold_sauvola(corrected, window_size=window_size)
        sauvola_binary = (corrected > sauvola_thresh).astype(np.uint8) * 255
        variants["sauvola"] = sauvola_binary
    else:
        variants["sauvola"] = None  # scikit-image not installed
 
    return variants
 
 
def correct_orientation(img, min_confidence=1.0):
    """
    Detects whether the flattened page is upside down or sideways using
    Tesseract's Orientation and Script Detection (OSD, --psm 0), and
    rotates it upright. Returns (corrected_image, degrees_rotated).
 
    Safe by design: OSD needs a reasonable amount of text and can fail
    on sparse/blurry pages. If it raises an error, or its confidence is
    below `min_confidence`, the image is returned unchanged (0 degrees)
    rather than risking a wrong rotation.
    """
    try:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        osd = pytesseract.image_to_osd(gray, config="--psm 0", output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))              # degrees to rotate clockwise to fix
        confidence = float(osd.get("orientation_conf", 0))
    except Exception:
        return img, 0
 
    if angle == 0 or confidence < min_confidence:
        return img, 0
 
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180), 180
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 270
    return img, 0
 
 
def ocr_image(img):
    """Runs Tesseract on a single-channel or BGR image, returns text."""
    if img is None:
        return ""
    return pytesseract.image_to_string(img, config=f"--psm {OCR_PSM}")
 
 
# =========================================================
# Evaluation: Character Error Rate (CER) / Word Error Rate (WER)
# =========================================================
 
def _levenshtein(seq_a, seq_b):
    """Standard edit-distance DP, works on a list of chars or words."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[m]
 
 
def compute_cer_wer(hypothesis, reference):
    """
    CER = edit distance at character level / reference length * 100
    WER = edit distance at word level / reference word count * 100
    Both hypothesis and reference are raw OCR/ground-truth strings.
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
 
    if len(ref_chars) == 0:
        cer = 0.0 if len(hyp_chars) == 0 else 100.0
    else:
        cer = _levenshtein(hyp_chars, ref_chars) / len(ref_chars) * 100
 
    if len(ref_words) == 0:
        wer = 0.0 if len(hyp_words) == 0 else 100.0
    else:
        wer = _levenshtein(hyp_words, ref_words) / len(ref_words) * 100
 
    return round(cer, 2), round(wer, 2)
 
 
# =========================================================
# Core per-image pipeline
# =========================================================
 
def scan_document(input_path, output_dir="."):
    """
    Runs Phases 1-4 on a single image. Saves the corner-detection debug
    image, the flattened color image, and each binarization variant.
    Returns a dict with file paths, OCR text per variant, and the raw
    (unprocessed) OCR text for comparison.
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image at {input_path}")
        return None
 
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    img_out_dir = os.path.join(output_dir, base_name)
    os.makedirs(img_out_dir, exist_ok=True)
 
    orig_h, orig_w = image.shape[:2]
 
    # --- Phase 1: small copy for fast, stable contour detection ---
    proc_width = 800
    ratio = orig_w / float(proc_width)
    proc_h = int(orig_h / ratio)
    small = cv2.resize(image, (proc_width, proc_h))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
 
    # --- Phase 2: find the 4 corners ---
    corners_small, method = find_document_contour(gray_small)
    if corners_small is None:
        print(f"Failed: Could not find document boundaries in {input_path}.")
        return None
 
    corners_full = corners_small * ratio
 
    debug_img = image.copy()
    cv2.drawContours(debug_img, [corners_full.astype(int)], -1, (0, 255, 0), 8)
    for (x, y) in corners_full.astype(int):
        cv2.circle(debug_img, (x, y), 15, (0, 0, 255), -1)
    debug_path = os.path.join(img_out_dir, "1_detected_corners.jpg")
    cv2.imwrite(debug_path, debug_img)
 
    # --- Phase 3: perspective warp on the FULL-RES image ---
    flattened = warp_perspective(image, corners_full)
 
    # --- Orientation fix: rotate upright if the page is upside down / sideways ---
    flattened, rotation_applied = correct_orientation(flattened)
 
    flattened_path = os.path.join(img_out_dir, "2_flattened_color.jpg")
    cv2.imwrite(flattened_path, flattened)
 
    flat_gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
 
    # --- Phase 4: binarization variants + OCR ---
    variants = binarize_variants(flat_gray)
    variant_paths = {}
    variant_texts = {}
    for name, img in variants.items():
        if img is None:
            variant_texts[name] = None
            continue
        path = os.path.join(img_out_dir, f"3_binarized_{name}.jpg")
        cv2.imwrite(path, img)
        variant_paths[name] = path
        variant_texts[name] = ocr_image(img)
 
    # Also OCR the plain flattened grayscale (no binarization) for reference
    variant_texts["grayscale_only"] = ocr_image(flat_gray)
 
    # Raw, unprocessed photograph OCR (the synopsis's "before" baseline)
    raw_text = ocr_image(image)
 
    # Pick the variant with the most extracted characters as the "best" text
    # (used only when no ground truth is available to pick by CER instead)
    best_name = max(variant_texts, key=lambda k: len(variant_texts[k].strip()) if variant_texts[k] else 0)
    best_text = variant_texts[best_name] or ""
 
    text_path = os.path.join(img_out_dir, "extracted_text.txt")
    with open(text_path, "w") as f:
        f.write(best_text)
 
    return {
        "base_name": base_name,
        "debug_path": debug_path,
        "flattened_path": flattened_path,
        "variant_paths": variant_paths,
        "variant_texts": variant_texts,
        "raw_text": raw_text,
        "best_variant": best_name,
        "best_text": best_text,
        "text_path": text_path,
        "method": method,
        "rotation_applied": rotation_applied,
    }
 
 
# =========================================================
# Batch driver + evaluation report
# =========================================================
 
def find_ground_truth(base_name, gt_dir):
    if not gt_dir:
        return None
    candidate = os.path.join(gt_dir, base_name + ".txt")
    if os.path.exists(candidate):
        with open(candidate, "r") as f:
            return f.read()
    return None
 
 
def _open_in_viewer(path):
    """Best-effort: pop the image open in the OS's default viewer.
    Windows only (uses os.startfile); silently does nothing elsewhere
    or if it fails, since this is just a convenience, not core logic."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: this only exists on Windows
    except Exception:
        pass
 
 
def run_pipeline(input_path, output_dir=".", ground_truth_dir=None, show=False):
    if os.path.isdir(input_path):
        image_paths = sorted([
            p for p in glob.glob(os.path.join(input_path, "*"))
            if p.lower().endswith(IMAGE_EXTENSIONS)
        ])
    else:
        image_paths = [input_path]
 
    if not image_paths:
        print(f"No images found at {input_path}")
        return
 
    os.makedirs(output_dir, exist_ok=True)
    report_rows = []
 
    for path in image_paths:
        print(f"\n=== Processing {os.path.basename(path)} ===")
        result = scan_document(path, output_dir)
        if result is None:
            continue
 
        print(f"Corners found via: {result['method']}")
        print(f"Orientation fix: rotated {result['rotation_applied']} degrees"
              if result["rotation_applied"] else "Orientation fix: none needed")
        for name, text in result["variant_texts"].items():
            n_chars = len(text.strip()) if text else 0
            print(f"  OCR [{name}]: {n_chars} characters extracted")
        print(f"Selected best variant: {result['best_variant']}")
 
        if show:
            _open_in_viewer(result["debug_path"])
            _open_in_viewer(result["flattened_path"])
 
        gt_text = find_ground_truth(result["base_name"], ground_truth_dir)
        if gt_text is not None:
            row_variants = {"raw_photograph": result["raw_text"]}
            row_variants.update(result["variant_texts"])
            for variant_name, hyp_text in row_variants.items():
                if hyp_text is None:
                    continue
                cer, wer = compute_cer_wer(hyp_text, gt_text)
                report_rows.append({
                    "image": result["base_name"],
                    "input_type": variant_name,
                    "CER (%)": cer,
                    "WER (%)": wer,
                })
            print(f"Ground truth found -- CER/WER computed for {result['base_name']}")
        else:
            if ground_truth_dir:
                print(f"No ground-truth file for {result['base_name']} "
                      f"(expected {result['base_name']}.txt in {ground_truth_dir})")
 
    if report_rows:
        report_path = os.path.join(output_dir, "evaluation_report.csv")
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "input_type", "CER (%)", "WER (%)"])
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\nEvaluation report written to: {report_path}")
 
        # Print a readable summary table (avg CER/WER per input_type across all images)
        from collections import defaultdict
        agg = defaultdict(list)
        for row in report_rows:
            agg[row["input_type"]].append((row["CER (%)"], row["WER (%)"]))
 
        print("\n--- Average CER / WER by input type (Table 1) ---")
        print(f"{'Input type':<22}{'CER (%)':>10}{'WER (%)':>10}")
        for input_type, vals in agg.items():
            avg_cer = round(sum(v[0] for v in vals) / len(vals), 2)
            avg_wer = round(sum(v[1] for v in vals) / len(vals), 2)
            print(f"{input_type:<22}{avg_cer:>10}{avg_wer:>10}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated OCR Document Scanner")
    parser.add_argument("input_path", help="Path to a single image, or a directory for batch mode")
    parser.add_argument("output_dir", nargs="?", default=".", help="Where to write results")
    parser.add_argument("--ground-truth", dest="ground_truth_dir", default=None,
                         help="Directory of .txt files (same base filename as each image) "
                              "for CER/WER evaluation")
    parser.add_argument("--psm", type=int, default=6,
                         help="Tesseract page segmentation mode (default 6: single block; "
                              "try 3 for multi-column full pages)")
    parser.add_argument("--show", action="store_true",
                         help="Pop open the detected-corners and flattened images in your "
                              "default photo viewer after processing each file (Windows only)")
    args = parser.parse_args()
    OCR_PSM = args.psm
 
    run_pipeline(args.input_path, args.output_dir, args.ground_truth_dir, args.show)
 
