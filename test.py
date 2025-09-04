from pdf2image import convert_from_path

pdf_path = r"C:\Users\saran\OneDrive\Desktop\GENAIEXHANGEHACKATHON\file-example_PDF_500_kB.pdf"
poppler_path = r"C:\Users\saran\Release-25.07.0-0\poppler-25.07.0\Library\bin"  # adjust this to your Poppler location

pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
for i, page in enumerate(pages):
    page.save(f"page_{i+1}.png", "PNG")