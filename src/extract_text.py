import fitz  # pymupdf
import io
import base64
import streamlit as st
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def extract_documents_from_pdf(file_path: str, source_name: str):
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    doc = fitz.open(file_path)
    documents = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        
        # 1. 텍스트 레이어 즉시 추출
        page_text = page.get_text().strip()
        
        # 2. 페이지 내 이미지 객체 찾기
        image_list = page.get_images(full=True)
        image_descriptions = []

        if image_list:
            st.toast(f"🎨 {page_number + 1}p: 그림 {len(image_list)}개 분석 중...")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Gemini에게 개별 이미지 분석 요청
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                }
                text_message = {
                    "type": "text",
                    "text": "이 그림/차트가 무엇을 설명하는지 핵심만 한두 문장으로 요약해줘."
                }
                
                try:
                    res = vision_model.invoke([HumanMessage(content=[text_message, image_message])])
                    image_descriptions.append(f"[그림{img_index+1} 설명: {res.content}]")
                except:
                    continue

        # 3. 텍스트와 그림 설명 결합
        full_content = f"{page_text}\n\n" + "\n".join(image_descriptions)
        
        documents.append(
            Document(
                page_content=full_content,
                metadata={"source": source_name, "page": page_number}
            )
        )

    doc.close()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_documents(documents)