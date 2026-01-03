import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from extract_text import extract_text_from_pdf, split_text  # 아까 만든 함수 재사용

# 1. 환경 변수 로드
load_dotenv()

def create_vector_db():
    # PDF 경로 설정
    pdf_path = os.path.join("data", "Test.pdf")
    
    # 2. 텍스트 추출 및 쪼개기
    print("PDF에서 텍스트를 추출하고 있습니다...")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(raw_text)
    
    # 3. Gemini 임베딩 모델 설정
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4. 벡터 저장소(Vector DB) 생성 및 저장
    print(f"{len(chunks)}개의 데이터 조각을 숫자로 변환하여 저장 중입니다...")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    
    # 5. 로컬 폴더로 저장 (나중에 다시 불러올 수 있게)
    vector_db.save_local("my_vector_db")
    print("저장이 완료되었습니다! 'my_vector_db' 폴더가 생성되었습니다.")

if __name__ == "__main__":
    create_vector_db()