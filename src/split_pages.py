from PyPDF2 import PdfReader, PdfWriter
import os

input_pdf = r"D:\Astro Books\Books\BPHS-hindi-compressed.pdf"
pages_per_file = 50

reader = PdfReader(input_pdf)
total_pages = len(reader.pages)

for start in range(0, total_pages, pages_per_file):
    writer = PdfWriter()

    end = min(start + pages_per_file, total_pages)

    for page_num in range(start, end):
        writer.add_page(reader.pages[page_num])

    output_file = f"BPHS.part_{start // pages_per_file + 1}.pdf"

    with open(output_file, "wb") as f:
        writer.write(f)

    # Get file size
    size_bytes = os.path.getsize(output_file)
    size_mb = size_bytes / (1024 * 1024)

    print(
        f"Created: {output_file} | "
        f"Pages: {start + 1}-{end} | "
        f"Size: {size_mb:.2f} MB"
    )

print("\nDone!")