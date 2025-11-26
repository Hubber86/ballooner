"""
balloon_generator.py
Pure-Python Balloon Diagram Generator
(Compatible with Python 3.14+, no OpenCV required)

Features:
- PDF → PNG conversion using PyMuPDF
- Image processing using Pillow + NumPy
- Custom Connected Component Labeling (no cv2)
- Balloon placement + leader lines
- OCR near detected parts
- Output: PNG + PDF + optional CSV report
"""

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import fitz  # PyMuPDF
import math
import uuid
import os
import csv

# ----------------------------------------------------------
# Data Classes
# ----------------------------------------------------------

@dataclass
class Part:
    id: int
    centroid: Tuple[int,int]
    bbox: Tuple[int,int,int,int]
    text: Optional[str] = None
    balloon_pos: Optional[Tuple[int,int]] = None

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def pdf_page_to_image(pdf_path: str, page_number: int = 0, dpi: int = 200) -> str:
    """Convert PDF page to PNG."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    out_path = f"{uuid.uuid4().hex}.png"
    pix.save(out_path)
    doc.close()
    return out_path


def save_image_as_pdf(image_path: str, output_pdf: str):
    """Save PIL image into a PDF."""
    img = Image.open(image_path).convert("RGB")
    img.save(output_pdf, "PDF", resolution=100.0)


# ----------------------------------------------------------
# Part Detection (Pure Python)
# ----------------------------------------------------------

def detect_parts_simple(image_path: str, min_area=80, max_area=30000) -> List[Part]:
    """
    Pure-Python connected-component detector.
    Steps:
      1) Convert to grayscale
      2) Threshold
      3) BFS label connected components
      4) Filter by area
      5) Compute centroids
    """
    pil = Image.open(image_path).convert("L")
    arr = np.array(pil)

    # Normalize + threshold
    thresh = (arr < 180).astype(np.uint8)   # dark = foreground

    H, W = thresh.shape
    visited = np.zeros_like(thresh, dtype=bool)

    parts = []
    pid = 1

    # 4-neighbor BFS
    def bfs(sx, sy):
        queue = [(sx, sy)]
        visited[sy, sx] = True
        pixels = [(sx, sy)]

        while queue:
            x, y = queue.pop()
            for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= nx < W and 0 <= ny < H:
                    if not visited[ny,nx] and thresh[ny,nx] == 1:
                        visited[ny,nx] = True
                        queue.append((nx, ny))
                        pixels.append((nx, ny))
        return pixels

    for y in range(H):
        for x in range(W):
            if thresh[y,x] == 1 and not visited[y,x]:
                pixels = bfs(x, y)
                area = len(pixels)
                if min_area <= area <= max_area:
                    xs = [p[0] for p in pixels]
                    ys = [p[1] for p in pixels]
                    cx = int(sum(xs)/area)
                    cy = int(sum(ys)/area)
                    bbox = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
                    parts.append(Part(pid, (cx,cy), bbox))
                    pid += 1

    # Sort top-to-bottom left-to-right
    parts.sort(key=lambda p: (p.centroid[1], p.centroid[0]))

    # Renumber cleanly
    for i,p in enumerate(parts, start=1):
        p.id = i

    return parts


# ----------------------------------------------------------
# OCR
# ----------------------------------------------------------

def ocr_text_near(image_path: str, center: Tuple[int,int], radius: int = 50) -> str:
    img = Image.open(image_path).convert("L")
    W, H = img.size
    x, y = center

    crop = img.crop((
        max(0, x-radius),
        max(0, y-radius),
        min(W, x+radius),
        min(H, y+radius)
    ))

    text = pytesseract.image_to_string(crop, config="--psm 6")
    return text.strip()


# ----------------------------------------------------------
# Balloon Placement
# ----------------------------------------------------------

def find_balloon_position(img_size, occupied, target, r=28):
    W, H = img_size
    cx, cy = target
    offsets = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
    dists = [60, 100, 150, 200]

    for d in dists:
        for ox, oy in offsets:
            bx = cx + ox*d
            by = cy + oy*d
            rect = (bx-r, by-r, bx+r, by+r)

            # Bounds check
            if rect[0] < 0 or rect[1] < 0 or rect[2] > W or rect[3] > H:
                continue

            # Collision check
            ok = True
            for oc in occupied:
                if not (rect[2] < oc[0] or rect[0] > oc[2] or rect[3] < oc[1] or rect[1] > oc[3]):
                    ok = False
                    break
            if ok:
                return (bx,by), rect

    # fallback: bottom-right
    bx = min(W-r-10, cx+150)
    by = min(H-r-10, cy+150)
    rect = (bx-r, by-r, bx+r, by+r)
    return (bx,by), rect


# ----------------------------------------------------------
# Rendering
# ----------------------------------------------------------

def draw_ballooned_image(image_path: str, parts: List[Part], out_path: str, r=28):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    occupied = []

    for p in parts:
        if not p.text:
            p.text = ocr_text_near(image_path, p.centroid)

        pos, rect = find_balloon_position((W,H), occupied, p.centroid, r)
        p.balloon_pos = pos
        occupied.append(rect)

        bx, by = pos
        cx, cy = p.centroid

        # leader line
        angle = math.atan2(cy-by, cx-bx)
        edge_x = bx + r * math.cos(angle)
        edge_y = by + r * math.sin(angle)
        mid = ((edge_x+cx)/2, (edge_y+cy)/2)
        draw.line([ (edge_x,edge_y), mid, (cx,cy) ], fill="red", width=2)

        # balloon circle
        draw.ellipse([bx-r, by-r, bx+r, by+r], outline="red", width=3)

        # number
        text = str(p.id)
        bbox = draw.textbbox((0,0), text, font=font)
        tw = bbox[2]-bbox[0]
        th = bbox[3]-bbox[1]
        draw.text((bx-tw/2, by-th/2), text, fill="red", font=font)

    # BOM box
    bom = [f"{p.id}. {p.text or '—'}" for p in parts]

    box_w = 300
    bbox_line = draw.textbbox((0,0), "A", font=small)
    line_h = (bbox_line[3]-bbox_line[1]) + 4
    box_h = line_h * len(bom) + 40

    x1, y1 = W-box_w-10, H-box_h-10
    x2, y2 = W-10, H-10

    draw.rectangle([x1,y1,x2,y2], fill=(240,240,240))
    draw.rectangle([x1,y1,x2,y1+28], fill=(200,200,200))
    draw.text((x1+8,y1+4), "Parts List (BOM)", font=font, fill="black")

    y = y1+32
    for line in bom:
        draw.text((x1+8,y), line, font=small, fill="black")
        y += line_h

    img.save(out_path)
    print("Saved:", out_path)


# ----------------------------------------------------------
# CSV Report
# ----------------------------------------------------------

def save_report_csv(parts: List[Part], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID","Centroid_X","Centroid_Y","BBox_X","BBox_Y","BBox_W","BBox_H","Balloon_X","Balloon_Y","Text"])
        for p in parts:
            bx, by = p.balloon_pos if p.balloon_pos else (None, None)
            writer.writerow([p.id, p.centroid[0], p.centroid[1],
                             p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3],
                             bx, by, p.text or ""])
    print("CSV report saved:", csv_path)


# ----------------------------------------------------------
# High-level API
# ----------------------------------------------------------

def generate_ballooned_diagram(input_path, out_img="ballooned.png", out_pdf=None, report_csv=None):
    if input_path.lower().endswith(".pdf"):
        img_path = pdf_page_to_image(input_path)
    else:
        img_path = input_path

    parts = detect_parts_simple(img_path)
    draw_ballooned_image(img_path, parts, out_img)

    if out_pdf:
        save_image_as_pdf(out_img, out_pdf)
        print("Saved:", out_pdf)

    if report_csv:
        save_report_csv(parts, report_csv)

    return parts


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image or PDF")
    parser.add_argument("--out-img", default="ballooned.png")
    parser.add_argument("--out-pdf", default=None)
    parser.add_argument("--report-csv", default=None)
    args = parser.parse_args()

    parts = generate_ballooned_diagram(args.input, args.out_img, args.out_pdf, args.report_csv)
    print("Detected parts:", len(parts))
    for p in parts:
        print(asdict(p))
