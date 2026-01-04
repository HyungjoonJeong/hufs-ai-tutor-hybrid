import fitz  # pymupdf
import io
import base64
import streamlit as st
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def extract_documents_from_pdf(file_path: str, source_name: str):
    vision_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    doc = fitz.open(file_path)
    documents = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        
        # 1. 텍스트는 원본에서 즉시 추출
        page_text = page.get_text().strip()
        
        # 2. 그림이 있는 경우에만 처리
        if page.get_images(full=True):
            st.toast(f"🖼️ {page_number + 1}p: 시각 자료 추출 중...")
            
            # 텍스트 영역을 모두 찾아서 흰색으로 가립니다 (Redact)
            for text_instance in page.search_for(" "): # 모든 공백/문자 탐색
                page.add_redact_annot(text_instance, fill=(1, 1, 1)) # 흰색 채우기
            page.apply_redactions() # 가리기 적용
            
            # 이제 텍스트가 사라진 '그림만 남은 페이지'를 이미지로 변환
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_data = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            
            # Gemini에게 순수하게 시각 정보만 분석 요청
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "이 이미지에서 글자는 무시하고, 그림이나 도표가 무엇을 의미하는지 분석해줘."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                ]
            )
            
            try:
                res = vision_model.invoke([message])
                page_text += f"\n\n[시각 자료 분석]\n{res.content}"
            except:
                pass

        documents.append(
            Document(
                page_content=page_text,
                metadata={"source": source_name, "page": page_number}
            )
        )

    doc.close()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(documents)