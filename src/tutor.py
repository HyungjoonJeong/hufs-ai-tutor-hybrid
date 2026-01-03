import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
# 기존 코드 (에러 발생)
# from langchain.prompts import PromptTemplate

# 수정된 코드 (최신 버전 경로)
from langchain_core.prompts import PromptTemplate

load_dotenv()

def run_ai_tutor():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local("my_vector_db", embeddings, allow_dangerous_deserialization=True)
    
    # 1. 모델 설정 (최신 버전 유지)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7) # 창의성을 위해 0.7로 조절

    # 2. 튜터 전용 지침(Prompt) 만들기
    template = """
    너는 한국외대 학생들을 위한 아주 친절하고 유능한 'AI 학습 튜터'야.
    아래 제공된 [문맥]을 바탕으로 학생의 [질문]에 답변해줘.
    
    답변 지침:
    1. 답변은 반드시 한국어로 해줘.
    2. 모르는 내용이라면 억지로 지며내지 말고 "강의 자료에 해당 내용이 없습니다"라고 솔직하게 말해줘.
    3. 학생이 이해하기 쉽게 핵심 요약을 먼저 해주고, 상세 설명을 덧붙여줘.
    4. 답변 마지막에는 항상 학생을 격려하는 한마디나, "더 궁금한 점이 있나요?"라는 질문을 남겨줘.

    [문맥]: {context}
    
    [질문]: {question}
    
    나의 답변:"""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    # 3. 질문-답변 엔진 조립 (프롬프트 추가)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT} # 프롬프트 연결!
    )

    print("\n[🎓 HUFS AI 튜터가 친절한 모드로 시작합니다!]")
    
    while True:
        query = input("\n학생 질문: ")
        if query == "나가기":
            break
        
        print("튜터가 자료를 분석 중입니다...")
        response = qa_chain.invoke(query)
        print(f"\nAI 튜터: {response['result']}")

if __name__ == "__main__":
    run_ai_tutor()