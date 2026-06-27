from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Step 1: Split by markdown headers → preserves chapter/section metadata
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "book"),
        ("##",  "chapter"),
        ("###", "section"),
    ],
    strip_headers=False
)

# Step 2: Further split large chunks on sloka/verse boundaries
char_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n**Sloka",   # BPHS / Phaladeepika sloka marker
        "\n**Verse",   # alternate marker
        "\n**Stanza",  # Saravali uses this sometimes
        "\n\n",        # paragraph break
        "\n",          # line break
        ". ",          # sentence
        ""             # character fallback
    ],
    chunk_size=1000,
    chunk_overlap=200,
    keep_separator=True
)


def split_documents(docs):
    final_chunks = []

    for doc in docs:
        # Stage 1: header-based split
        md_chunks = header_splitter.split_text(doc.page_content)

        for chunk in md_chunks:
            # carry over source filename into every chunk's metadata
            chunk.metadata.update(doc.metadata)

            # Stage 2: only re-split if chunk is too large
            if len(chunk.page_content) > 700:
                sub_chunks = char_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

    print(f'Total chunks created: {len(final_chunks)}')
    return final_chunks


if __name__ == "__main__":
    from document_loader import load_documents
    docs = load_documents()
    chunks = split_documents(docs)

    # inspect a sample
    print("\n--- Sample Chunk ---")
    print(chunks[50].page_content)
    print("\n--- Metadata ---")
    print(chunks[10].metadata)
