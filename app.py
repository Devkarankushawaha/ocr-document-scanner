"""
Live demo for the Automated OCR Document Scanner.

Run (from the folder containing scanner_ocr.py and this file):
    python -m streamlit run app.py
"""

import os
import tempfile

import cv2
import numpy as np

import scanner_ocr
from scanner_ocr import scan_document, compute_cer_wer


def process_bytes(data, ground_truth=None):
    """
    Runs the full scanner pipeline on uploaded image bytes.
    Returns a dict of everything the UI needs, or None if no document
    boundary could be found. Kept free of any Streamlit code so it can
    be tested on its own.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "upload.jpg")
        with open(in_path, "wb") as f:
            f.write(data)

        result = scan_document(in_path, tmp)
        if result is None:
            return None

        out = {
            "original": cv2.imread(in_path),
            "corners": cv2.imread(result["debug_path"]),
            "flattened": cv2.imread(result["flattened_path"]),
            "method": result["method"],
            "rotation": result["rotation_applied"],
            "best_variant": result["best_variant"],
            "best_text": result["best_text"],
            "variants": {},
            "metrics": [],
        }

        for name, text in result["variant_texts"].items():
            path = result["variant_paths"].get(name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
            out["variants"][name] = {"image": img, "text": text or ""}

        if ground_truth and ground_truth.strip():
            candidates = {"raw_photograph": result["raw_text"]}
            candidates.update({k: v["text"] for k, v in out["variants"].items()})
            for name, text in candidates.items():
                cer, wer = compute_cer_wer(text, ground_truth)
                out["metrics"].append({"Input type": name, "CER (%)": cer, "WER (%)": wer})

        ok, buf = cv2.imencode(".jpg", out["flattened"])
        out["flattened_jpg"] = buf.tobytes() if ok else b""
        return out


def main():
    import streamlit as st
    import pytesseract

    st.set_page_config(page_title="OCR Document Scanner", page_icon="📄", layout="wide")
    st.title("📄 Automated OCR Document Scanner")
    st.caption("Upload a photo of a printed document. The app finds the page, flattens it, "
               "cleans up lighting and reads the text.")

    with st.sidebar:
        st.header("Settings")
        tess_path = st.text_input(
            "Tesseract path (Windows only, if needed)",
            value="",
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
        if tess_path.strip():
            pytesseract.pytesseract.tesseract_cmd = tess_path.strip()
        psm = st.selectbox(
            "Text layout mode (Tesseract --psm)",
            [6, 3, 4, 11],
            help="6: one block (best for receipts and invoices). 3: automatic, "
                 "for multi-column pages. 4: single column. 11: sparse text.",
        )
        scanner_ocr.OCR_PSM = psm
        st.markdown("**Pipeline**\n\n1. Gray, blur, Canny\n2. Contours and 4 corners\n"
                    "3. Perspective warp and orientation fix\n4. Shadow removal, thresholding, OCR")

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        st.error("Tesseract OCR was not found. Install it, or paste the full path to "
                 "`tesseract.exe` in the sidebar.")
        st.stop()

    uploaded = st.file_uploader("Document photo", type=["jpg", "jpeg", "png"])
    ground_truth = st.text_area(
        "Optional: type the true text of the document to get CER / WER scores",
        height=100,
    )

    if uploaded is None:
        st.info("Upload a photo to begin.")
        return

    with st.spinner("Scanning..."):
        result = process_bytes(uploaded.getvalue(), ground_truth)

    if result is None:
        st.error("Could not find the document boundary. Try a photo where the page "
                 "contrasts with the background and all four corners are visible.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Corner detection", result["method"].split(" (")[0])
    c2.metric("Rotation fixed", f"{result['rotation']}°" if result["rotation"] else "none needed")
    c3.metric("Best binarization", result["best_variant"])

    st.subheader("Pipeline stages")
    s1, s2, s3 = st.columns(3)
    s1.image(result["original"], channels="BGR", caption="1. Original photo", use_container_width=True)
    s2.image(result["corners"], channels="BGR", caption="2. Detected corners", use_container_width=True)
    s3.image(result["flattened"], channels="BGR", caption="3. Flattened page", use_container_width=True)

    st.subheader("Binarization methods")
    names = [n for n, v in result["variants"].items() if v["image"] is not None]
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        with tab:
            left, right = st.columns(2)
            left.image(result["variants"][name]["image"], caption=name, use_container_width=True)
            right.text_area("OCR text", result["variants"][name]["text"], height=300, key=f"t_{name}")

    if result["metrics"]:
        st.subheader("Accuracy against your ground truth (lower is better)")
        st.dataframe(result["metrics"], use_container_width=True, hide_index=True)

    st.subheader("Download")
    d1, d2 = st.columns(2)
    d1.download_button("Extracted text (.txt)", result["best_text"],
                       file_name="extracted_text.txt", use_container_width=True)
    d2.download_button("Flattened scan (.jpg)", result["flattened_jpg"],
                       file_name="scanned_document.jpg", mime="image/jpeg",
                       use_container_width=True)


if __name__ == "__main__":
    main()
