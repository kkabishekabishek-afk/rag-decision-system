from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


if __name__ == "__main__":

    pdf_folder = Path("data/documents")

    pdf_files = list(
        pdf_folder.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF found in data/documents"
        )

        exit()

    pdf_path = pdf_files[0]

    pages = load_pdf(pdf_path)

    print(
        f"\nPDF: {pdf_path.name}"
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    for page in pages[:2]:

        print(
            f"\n--- Page {page['page']} ---"
        )

        print(
            page["text"][:1000]
        )