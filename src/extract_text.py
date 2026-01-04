from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader


def extract_documents_from_pdf(file_path: str, source_name: str):
    """
    PDF → LangChain Document 리스트로 변환
    (페이지 번호, 파일명 메타데이터 포함)
    """
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages):
        # OCR은 Gemini Vision이 담당 (가성비 최고)
        vision_model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
        text = page.extract_text()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": page_number
                }
            )
        )

    return documents


def split_documents(documents):
    """
    Document 리스트 → chunk 단위로 분할
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)
