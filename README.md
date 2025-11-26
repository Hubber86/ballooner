# 🟠 Ballooner — Automatic Balloon Diagram Generator (Python)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pillow.svg)](https://pypi.org/project/Pillow/)

**Ballooner** is a pure-Python tool that automatically generates **ballooned engineering diagrams** from PDFs or images. It detects parts, numbers them with balloons, draws leader lines, extracts nearby text using OCR, and outputs a **PNG image**, **PDF**, and **Parts List (BOM)**.

---

## 📌 Features

- ✅ **Automatic Part Detection**  
  Pure-Python connected-component labeling; no OpenCV required.  

- 🗨️ **Balloon Numbering & Placement**  
  Smart placement avoids overlapping balloons; adjustable radius.  

- 📝 **Text Extraction (OCR)**  
  Uses Tesseract OCR to capture part labels for the BOM.  

- 📄 **Output**  
  - Ballooned diagram → PNG  
  - Optional PDF export  
  - BOM overlay included in the image  

- ⚡ Lightweight, fast, and dependency-minimal.

---

## 🖼️ Screenshots

![Sample Ballooned Diagram](out.png)  
*Example of an auto-generated ballooned diagram.*

---

## 🧰 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Hubber86/ballooner.git
cd ballooner
2. Install Python dependencies
pip install pymupdf pillow numpy pytesseract
3. Install Tesseract OCR
Windows: Download Tesseract Installer

Linux: sudo apt install tesseract-ocr

macOS: brew install tesseract

Make sure Tesseract executable is added to your PATH or update the script path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

🚀 Usage
CLI
python balloon_generator.py "Input.pdf" --out-img out.png --out-pdf out.pdf
Output
Detected parts: 14
Saved: out.png
Saved: out.pdf
--out-img : output ballooned image (PNG)

--out-pdf : optional ballooned PDF

📁 Project Structure
ballooner/
│
├── balloon_generator.py   # Main script
├── README.md
├── examples/
│   ├── sample_diagram.png
│   └── sample_ballooned.png
└── output/
    ├── ballooned.png
    └── ballooned.pdf
💡 Why Ballooner?
Creating balloon diagrams manually is time-consuming for QA, assembly, or inspection documentation. Ballooner automates the process, reducing errors and speeding up engineering workflows.

🛠️ Future Enhancements
ML-based part segmentation

CAD-aware symbol detection

Multi-page PDF support

Export BOM to CSV/Excel

Interactive web UI

📜 License
This project is licensed under the MIT License — free to use, modify, and distribute.


🔗 References
PyMuPDF Documentation

Pillow Documentation

Tesseract OCR
