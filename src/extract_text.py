import fitz  # pymupdf
import io
import base64
import streamlit as st
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def extract_documents_from_pdf(file_path: str, source_name: str):
    # Gemini 모델 설정
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    doc = fitz.open(file_path)
    documents = []
    total = len(doc)

    for page_number in range(total):
        page = doc[page_number]
        
        # 1. 먼저 해당 페이지의 텍스트를 추출해봅니다.
        raw_text = page.get_text().strip()
        
        # [판단 로직] 
        # 텍스트가 150자 이상 풍부하게 있다면 -> 일반 텍스트 추출 모드 (초고속)
        # 텍스트가 거의 없다면 -> 이미지/스캔본으로 판단하고 OCR 모드 (정밀)
        if len(raw_text) > 150:
            st.toast(f"⚡ {page_number + 1}p: 텍스트 직독 중...")
            page_content = raw_text
        else:
            st.toast(f"👁️ {page_number + 1}p: 이미지 정밀 분석 중...")
            # 페이지를 이미지로 변환
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            encoded_image = base64.b64encode(img_data).decode("utf-8")
            
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            }
            text_message = {
                "type": "text",
                "text": "이 페이지는 이미지나 표 위주입니다. 내용을 아주 상세하게 마크다운 형식의 텍스트로 설명해줘."
            }
            
            try:
                response = vision_model.invoke([HumanMessage(content=[text_message, image_message])])
                page_content = response.content
            except Exception as e:
                page_content = f"OCR 실패: {raw_text if raw_text else str(e)}"

        documents.append(
            Document(
                page_content=page_content,
                metadata={"source": source_name, "page": page_number}
            )
        )

    doc.close()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(documents)