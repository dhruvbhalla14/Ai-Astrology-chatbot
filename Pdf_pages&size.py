print(f"Saved to: {output_file}")
import os
fileee = r'D:\Astro Books\Books\BPHS_hindi_part_2.pdf'
print(f"Size: {os.path.getsize(fileee)/(1024*1024):.2f} MB")


from PyPDF2 import PdfReader

pdf_path = r"D:\Astro Books\Books\BPHS_hindi_part_2.pdf"
reader = PdfReader(pdf_path)
print("Total Pages:", len(reader.pages))