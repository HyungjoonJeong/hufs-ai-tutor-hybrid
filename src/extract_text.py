import fitz  # pymupdf
import io
import base64
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import streamlit as st

def extract_documents_from_pdf(file_path: str, source_name: str):
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    doc = fitz.open(file_path)
    documents = []
    total = len(doc)

    for page_number in range(total):
        for page_number in range(total):
        # UI 업데이트용 (함수 안에서 스트림릿 UI를 직접 건드림)
            st.toast(f"📄 {page_number + 1} / {total} 페이지 분석 중...")

        page = doc[page_number]
        text = page.get_text().strip()
        
        # [전략] 텍스트가 일정량(예: 100자) 이상 있고, 이미지가 적으면 바로 텍스트 추출
        # 그렇지 않으면(이미지 PDF거나 표가 많으면) Gemini OCR 가동
        if len(text) > 100:
            page_content = f"[Text Extraction]\n{text}"
            st.toast(f"⚡ {page_number + 1}p: 텍스트 직독 중...")
        else:
            st.toast(f"👁️ {page_number + 1}p: 이미지 분석(OCR) 중...")
            # 고해상도 이미지 변환
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            encoded_image = base64.b64encode(img_data).decode("utf-8")
            
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            }
            text_message = {
                "type": "text",
                "text": "이 페이지의 내용을 아주 상세하게 텍스트로 복원해줘. 표는 마크다운으로, 그림은 설명으로 포함해줘."
            }
            
            try:
                response = vision_model.invoke([HumanMessage(content=[text_message, image_message])])
                page_content = f"[OCR Extraction]\n{response.content}"
            except Exception as e:
                page_content = f"에러 발생: {str(e)}"

        documents.append(
            Document(
                page_content=page_content,
                metadata={"source": source_name, "page": page_number}
            )
        )

    doc.close()
    return documents

def split_documents(documents):
    # OCR로 생성된 텍스트는 정보 밀도가 높으므로 chunk 크기를 넉넉하게 잡습니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)