# src/image_extraction.py

import os
from pathlib import Path
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageStat
import pytesseract


BASE_OUTPUT_DIR = Path(r"C:\Users\KAMALESH MUKHERJEE\Desktop\Multimodal Medical Report Analyzer\output")


# ------------------------------
# Detect file type
# ------------------------------
def detect_file_type(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        return "image"
    elif ext == ".dcm":
        return "dicom"
    return "unknown"


# ------------------------------
# SMART FILTER — Remove logos / icons / junk
# ------------------------------
def is_potential_medical_image(path):
    try:
        img = Image.open(path).convert("RGB")
    except:
        return False

    w, h = img.size

    # 1) Reject tiny images (most logos)
    if w < 400 or h < 400:
        return False

    # 2) Reject ultra colorful images (medical images rarely colorful)
    stat = ImageStat.Stat(img)
    r, g, b = stat.mean
    max_color_diff = max(abs(r - g), abs(g - b), abs(r - b))

    # High color variation → likely logo
    if max_color_diff > 40:
        return False

    # 3) Reject overly bright images (white logos)
    if max(r, g, b) > 240:
        return False

    # 4) Reject overly dark images (solid black shapes)
    if min(r, g, b) < 15:
        return False

    # 5) Reject weird aspect ratios
    ratio = w / h
    if ratio > 2.5 or ratio < 0.3:
        return False

    return True


# ------------------------------
# Extract raw images from PDF
# ------------------------------
def extract_images_from_pdf(pdf_path):
    output_dir = BASE_OUTPUT_DIR / "page_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    extracted = []

    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)

        for idx, img_info in enumerate(images):
            xref = img_info[0]
            base_img = doc.extract_image(xref)

            img_bytes = base_img["image"]
            ext = base_img["ext"]

            img_name = f"{pdf_path.stem}_page{page_num+1}_img{idx+1}.{ext}"
            img_path = output_dir / img_name

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            extracted.append(str(img_path))

    doc.close()
    return extracted


# ------------------------------
# OCR to detect full report pages
# ------------------------------
def is_report_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    text = pytesseract.image_to_string(img)
    return len(text.strip()) > 30


# ------------------------------
# Crop X-ray-like objects from reports
# ------------------------------
def crop_xrays_from_image(image_path):
    output_dir = BASE_OUTPUT_DIR / "xray_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    count = 1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 200 or h < 200:
            continue

        ratio = w / float(h)
        if ratio < 0.3 or ratio > 3.5:
            continue

        crop = img[y:y+h, x:x+w]
        crop_path = output_dir / f"{image_path.stem}_crop{count}.png"
        cv2.imwrite(str(crop_path), crop)

        crops.append(str(crop_path))
        count += 1

    return crops


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def process_file(path):
    file_type = detect_file_type(path)
    print(f"[INFO] Detected file type: {file_type}")

    # ---------------- PDF ----------------
    if file_type == "pdf":
        print("[INFO] Extracting images from PDF...")
        extracted = extract_images_from_pdf(path)

        all_crops = []
        for img in extracted:
            if is_report_image(img):
                print(f"[INFO] Cropping X-rays from report page: {img}")
                all_crops.extend(crop_xrays_from_image(img))
            else:
                all_crops.append(img)

        # 🔥 Apply medical filter here
        filtered = [im for im in all_crops if is_potential_medical_image(im)]
        return filtered

    # ---------------- IMAGE ----------------
    elif file_type == "image":
        if is_report_image(path):
            print("[INFO] Cropping X-rays from report...")
            crops = crop_xrays_from_image(path)
            return [c for c in crops if is_potential_medical_image(c)]
        else:
            return [path] if is_potential_medical_image(path) else []

    # ---------------- DICOM ----------------
    elif file_type == "dicom":
        print("[INFO] DICOM support to be added later.")
        return []

    else:
        raise ValueError("Unsupported file format")