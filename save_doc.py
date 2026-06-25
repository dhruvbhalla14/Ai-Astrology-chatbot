output_file = r"D:\Astro Books\BPHS_extracted.md"
with open(output_file, "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(doc.page_content)
        f.write("\n\n")

print(f"Saved to: {output_file}")