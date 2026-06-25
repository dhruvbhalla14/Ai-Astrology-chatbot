import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from llama_parse import LlamaParse
from PyPDF2 import PdfReader
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise RuntimeError("LLAMA_CLOUD_API_KEY is not set. Add it to .env or your shell environment.")

pdf_path = Path(r"D:\Astro Books\Books\Phaladeepika_part_2.pdf")
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

parser = LlamaParse(
    api_key=api_key,
    result_type="markdown",
    ignore_errors=False,
    max_timeout=1200
)
print("API Key:", api_key[:10] + "...")
print("PDF:", pdf_path)
print("PDF Size:", pdf_path.stat().st_size / (1024*1024), "MB")
print("Total Pages:", len(PdfReader(str(pdf_path)).pages))
print("Starting upload...")


llama_docs = parser.load_data(str(pdf_path))

docs = [Document(page_content=doc.text) for doc in llama_docs]

print(f"Loaded {len(docs)} documents")


output_file = r"D:\Astro Books\Books\Phaladeepika_part_2.md"
with open(output_file, "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(doc.page_content)
        f.write("\n\n")



print(f"Saved to: {output_file}")
output_file = r'D:\Astro Books\Books\Phaladeepika_part_2.md'
print(f"Size: {os.path.getsize(output_file)/(1024*1024):.2f} MB")

