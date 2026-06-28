from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter

input_pdf = Path(r"D:\Astro Books\Books\BPHS-hindi-compressed.pdf")

if not input_pdf.exists():
    raise FileNotFoundError(f"PDF not found: {input_pdf}")

reader = PdfReader(str(input_pdf))
total_pages = len(reader.pages)
split_at = (total_pages + 1) // 2

BPHS_hindi_part1 = input_pdf.with_name(f"{input_pdf.stem}_part_1.pdf")
BPHS_hindi_part2 = input_pdf.with_name(f"{input_pdf.stem}_part_2.pdf")


def save_part(start_page: int, end_page: int, output_path: Path) -> None:
    writer = PdfWriter()

    for page_num in range(start_page, end_page):
        writer.add_page(reader.pages[page_num])

    with open(output_path, "wb") as file:
        writer.write(file)


save_part(0, split_at, BPHS_hindi_part1)
save_part(split_at, total_pages, BPHS_hindi_part2)

print(f"Total pages: {total_pages}")
print(f"Part 1: pages 1 to {split_at} -> {BPHS_hindi_part1}")
print(f"Part 2: pages {split_at + 1} to {total_pages} -> {BPHS_hindi_part2}")
