from pdf2image import convert_from_path
from surya.ocr import run_ocr
from surya.model.ocr.model import load_model
from surya.model.ocr.processor import load_processor

PDF_PATH = r"D:\Astro Books\Books\BPHS-hindi-compressed.pdf"
OUTPUT_FILE = r"D:\Astro Books\Books\BPHS-hindiii-compressed.md"

# TEST MODE
FIRST_PAGE = 1
LAST_PAGE = 5

print("Loading Surya OCR model...")
model = load_model()
processor = load_processor()

print("Converting PDF pages to images...")
pages = convert_from_path(
    PDF_PATH,
    dpi=300,
    first_page=FIRST_PAGE,
    last_page=LAST_PAGE
)

print(f"Loaded {len(pages)} pages")

langs = [["hi"]] * len(pages)

print("Running OCR...")
predictions = run_ocr(
    pages,
    langs,
    model,
    processor
)

print("Saving output...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for page_num, page in enumerate(predictions, start=FIRST_PAGE):
        f.write(f"\n\n# PAGE {page_num}\n\n")

        for line in page.text_lines:
            f.write(line.text + "\n")

print(f"Done! Saved to: {OUTPUT_FILE}")



