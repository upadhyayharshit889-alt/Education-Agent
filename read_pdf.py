from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print("Number of pages:", len(documents))

text = documents[0].page_content

print("Characters found:", len(text))
print("\nPDF TEXT:")
print(text)