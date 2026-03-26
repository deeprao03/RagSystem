from app.loaders.pdf_loader import get_extracted_text_preview, load_pdf_documents


def run_step2_demo(pdf_path: str) -> None:
    documents = load_pdf_documents(pdf_path)
    print(f"Loaded {len(documents)} pages from: {pdf_path}")

    preview = get_extracted_text_preview(documents)
    print("\n--- Extracted Text Preview ---\n")
    print(preview if preview.strip() else "No text extracted.")


if __name__ == "__main__":
    sample_pdf_path = "data/raw/sample.pdf"
    run_step2_demo(sample_pdf_path)
