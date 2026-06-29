import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BOOK_NAMES = {
    "BPHS_English_Book.md":   "Brihat Parashara Hora Shastra (English)",
    "BPHS_Hindi_Book.md":     "Brihat Parashara Hora Shastra (Hindi)",
    "Phaladeepika_part_1.md": "Phaladeepika Part 1",
    "Phaladeepika_part_2.md": "Phaladeepika Part 2",
    "Saravali.md":            "Saravali",
}

char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    separators=["\n**Sloka", "\n**Verse", "\n**Stanza", "\n\n", "।", ". ", "? ", "! ", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=75,
    keep_separator=True
)

def remove_toc(text: str) -> str:
    match = re.search(r'\n# Chapter', text)
    if match:
        return text[match.start():]
    return text

def clean_text(text: str) -> str:
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove long OCR garbage tokens
    text = re.sub(r"\b[A-Za-z0-9-]{12,}\b", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()

def split_by_chapters(text: str):
    pattern = r'(?=# Chapter [\dIVXivx]+)'  # fixed: handles roman numerals too
    chunks = re.split(pattern, text)
    return [c.strip() for c in chunks if len(c.strip()) > 300]  # filter short chunks here

def split_documents(docs):
    final_chunks = []

    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", ""))
        book_name = BOOK_NAMES.get(filename, filename)

        text = clean_text(doc.page_content)
        text = remove_toc(text)

        chapter_chunks = split_by_chapters(text)

        for chapter_text in chapter_chunks:
            first_line = chapter_text.split('\n')[0].strip()
            chapter_title = first_line.replace('#', '').strip()

            chunk_doc = Document(
                page_content=chapter_text,
                metadata={
                    "book": book_name,
                    "chapter": chapter_title,
                    "source": doc.metadata.get("source", "")
                }
            )

            if len(chapter_text) > 700:
                sub_chunks = char_splitter.split_documents([chunk_doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk_doc)

    print(f'Total chunks created: {len(final_chunks)}')
    return final_chunks


if __name__ == "__main__":
    from document_loader import load_documents
    docs = load_documents()
    chunks = split_documents(docs)

    print("\n--- Sample Chunk ---")
    print(chunks[50].page_content)
    print("\n--- Metadata ---")
    print(chunks[10].metadata)