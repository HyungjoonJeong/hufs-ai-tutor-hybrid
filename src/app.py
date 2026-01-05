import streamlit as st
import os
import tempfile
import concurrent.futures # 파일 최상단에 추가해주세요!
from dotenv import load_dotenv

# 1. 핵심 설계도 (Core)
from langchain_core.prompts import PromptTemplate

# 2. 구글 AI 연결 (Google GenAI)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# 3. 데이터베이스 (Community)
from langchain_community.vectorstores import FAISS

# 4. 메모리 (가장 안전한 최신 경로로 변경)
# 만약 여기서 에러가 나면 from langchain_community.chat_message_histories import ... 로 선회해야 합니다.
#try:
#    from langchain.memory import ConversationBufferMemory
#except ImportError:
#    from langchain_community.memory import ConversationBufferMemory

# 5. 검색 엔진 (Classic)
from langchain_classic.chains import RetrievalQA

# 6. 직접 만든 모듈
from extract_text import extract_documents_from_pdf, split_documents

# --------------------------------
# 기본 설정
# --------------------------------
load_dotenv()

st.set_page_config(
    page_title="HUFS AI Tutor",
    layout="wide"
)

st.title("HUFS RAG 기반 AI 튜터 (GPT-5.2 & Gemini 2.5)")
st.caption("강의 자료 기반으로 GPT와 Gemini를 종합하여 답변하며 출처를 명확히 제시합니다.")

# --------------------------------
# 세션 상태
# --------------------------------
# --- app.py 상단 세션 상태 초기화 부분 ---

if "gpt_messages" not in st.session_state:
    st.session_state.gpt_messages = []

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []

# 기존 messages는 더 이상 쓰지 않지만, 
# 혹시 모르니 남겨두거나 아래처럼 깔끔하게 정리하세요.
# if "messages" not in st.session_state:
#     st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --------------------------------
# 질문 분류기
# --------------------------------
def classify_question(question: str) -> str:
    # 텍스트 분류는 설정이 복잡한 Gemini 대신 GPT-4o-mini를 씁니다. (매우 저렴)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"다음 질문을 'concept', 'calculation', 'summary' 중 하나로 분류해. 한 단어만 답해. 질문: {question}"
    result = llm.invoke(prompt)
    return result.content.strip().lower()


# --------------------------------
# --------------------------------
# 계산 문제 전용 체인 (GPT/Gemini 대응)
# --------------------------------
def run_calculation_chain(question: str, model_type: str, vector_db):
    # 1. 모델 선택
    if model_type == "gpt":
        llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    else:
        # 2026년 기준 최신 안정 버전인 1.5-flash 권장
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 2. 관련 문서 검색
# 함수 내부에서 st.session_state.vector_db 대신 vector_db 사용!
    docs = vector_db.similarity_search(question, k=7)
    context = "\n\n".join([d.page_content for d in docs])

    template = """
너는 대학 과목 계산 문제를 푸는 조교이다. 제공된 [문맥]의 공식과 수치를 바탕으로 문제를 풀어라.

[규칙]
1. 풀이 과정을 가독성을 위해 단계별로 번호를 매겨 반드시 숫자 인덱스(1., 2., 3.)를 사용하여 답변을 구조화하여 상세히 설명하라.
2. 수식은 LaTeX 형식이나 명확한 기호를 사용하여 제시하라.
3. 마지막에 최종 답을 '정답: '과 함께 정리하라.
4. 문맥에 없는 정보는 가급적 사용하지 말고, 데이터가 부족하면 문맥을 참고하라고 안내하라.
2. 각 포인트마다 핵심 키워드를 볼드체(**)로 표시하라.
3. 강의 자료의 내용을 구체적으로 인용하되, 문장은 자연스럽게 다듬어라.
5. 없는 내용은 추측하지 마라.
6. 마지막에 참고 자료와 출처를 명시하라.
7. 답변은 최소 3문단 이상의 충분한 분량으로 작성할 것.
8. 강의 자료에 있는 예시를 적극적으로 인용할 것.
9. 마지막에는 학습을 돕기 위해 '관련하여 추가로 알면 좋은 개념'을 두세 문장 덧붙일 것.

[문맥]
{context}

[문제]
{question}

[풀이]
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # 3. 답변 생성
    response = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )

    return response.content, docs
# --------------------------------
# 일반 RAG 체인
# --------------------------------
# 2. 메인 답변 체인 (GPT-4o 사용 - 정밀한 논리)
# run_rag 정의 부분 수정
def run_rag_final(question: str, answer_style: str, model_type: str, chat_history: list, docs: list):
    # 1. 모델 설정 (안정적인 모델명으로 수정)
    if model_type == "gpt":
        llm = ChatOpenAI(model="gpt-5.2", temperature=0.7)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # 2. 컨텍스트 구성
    context_text = "\n\n".join([d.page_content for d in docs])
    chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    
    length_instruction = (
        "핵심만 간결하게 답하라." if answer_style == "짧게" 
        else "자세히 설명하라."
    )

    template = """
당신은 한국외국어대학교의 1타 강사 AI 튜터입니다. 
제공된 [강의 자료]를 바탕으로 학생의 질문에 답변하세요.

[규칙]
1. 가독성을 위해 반드시 숫자 인덱스(1., 2., 3.)를 사용하여 답변을 구조화하라.
2. 각 포인트마다 핵심 제목을 볼드체(**)로 표시하라.
3. 강의 자료의 내용을 구체적으로 인용하되, 문장은 자연스럽게 다듬어라.
4. 반드시 문맥에 근거해 답하라.
5. 없는 내용은 추측하지 마라.
6. 마지막에 참고 자료와 출처를 명시하라.
7. 답변은 최소 3문단 이상의 충분한 분량으로 작성할 것.
8. 강의 자료에 있는 예시나 수치를 적극적으로 인용할 것.
9. 마지막에는 학습을 돕기 위해 '관련하여 추가로 알면 좋은 개념'을 두세 문장 덧붙일 것.
10. {length_instruction}

[이전 대화]
{chat_history}

[문맥]
{context}

[질문]
{question}

답변 (전문적이고 상세하게):
"""
    prompt = template.format(
        length_instruction=length_instruction,
        chat_history=chat_history_str,
        context=context_text,
        question=question
    )

    response = llm.invoke(prompt)
    return response.content



# --------------------------------
# 사이드바
# --------------------------------
# --------------------------------
# 사이드바
# --------------------------------
with st.sidebar:
    st.header("설정")

    answer_style = st.radio(
        "답변 길이",
        ["짧게", "자세히"],
        index=1
    )

    # 사이드바의 대화 초기화 버튼 부분
    if st.button("대화 초기화"):
        st.session_state.gpt_messages = []
        st.session_state.gemini_messages = []
        st.rerun()

    st.divider()

    uploaded_files = st.file_uploader(
        "PDF 업로드",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("학습 시작"):
        # --- 여기서부터 수정 (진행 바 및 텍스트 공간 확보) ---
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        with st.spinner("자료 분석 중..."):
            all_docs = []

            for i, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                # 🧐 현재 어떤 파일을 분석 중인지 표시
                status_placeholder.info(f"📄 '{file.name}' 분석 중... (파일 {i+1}/{len(uploaded_files)})")
                
                # 하이브리드 OCR 함수 호출
                docs = extract_documents_from_pdf(
                    tmp_path,
                    source_name=file.name
                )
                all_docs.extend(docs)
                os.remove(tmp_path)
                
                # 파일 단위로 진행률 업데이트
                progress_bar.progress(int((i + 1) / len(uploaded_files) * 50))

            status_placeholder.info("🧠 지식 데이터베이스 구축 중... (거의 다 됐어요!)")
            
            chunks = split_documents(all_docs)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )

            st.session_state.vector_db = FAISS.from_documents(
                chunks, embedding=embeddings
            )  

            # 모든 과정 완료 처리
            progress_bar.progress(100)
            status_placeholder.success(f"✅ 총 {len(uploaded_files)}개의 파일 학습 완료!")
            # --- 여기까지 수정 ---

# --------------------------------
# --------------------------------
# 채팅 UI (이전 대화 기록 복원)
# --------------------------------
# 화면을 2개로 나눠서 각 모델의 이전 대화 기록을 좌우에 배치합니다.
view_col1, view_col2 = st.columns(2)

with view_col1:
    for msg in st.session_state.gpt_messages:
        # 사용자의 질문과 GPT의 답변을 차례로 출력
        avatar = "🤖" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

with view_col2:
    for msg in st.session_state.gemini_messages:
        # 사용자의 질문과 Gemini의 답변을 차례로 출력
        avatar = "♊" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# --------------------------------
# 신규 질문 입력 및 처리 (병렬 & 공통 검색 버전)
# --------------------------------

if question := st.chat_input("질문을 입력하세요"):
    if st.session_state.vector_db is None:
        st.warning("먼저 PDF를 학습시켜주세요.")
    else:
        # 1. 공통 재료 준비 (메인 스레드에서 한 번만 검색!)
        # 이렇게 하면 스레드 내부 AttributeError와 구글 API 충돌을 완벽히 막습니다.
        with st.spinner("관련 강의 자료를 찾는 중..."):
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 7})
            shared_docs = retriever.invoke(question) # 공통 검색 결과

        gpt_h = st.session_state.gpt_messages.copy()
        gem_h = st.session_state.gemini_messages.copy()
        
        # 세션에 사용자 질문 즉시 저장
        st.session_state.gpt_messages.append({"role": "user", "content": question})
        st.session_state.gemini_messages.append({"role": "user", "content": question})

        # 화면에 사용자 질문 표시
        with st.chat_message("user"):
            st.markdown(question)

        # 2. 레이아웃 설정 및 질문 분류
        col1, col2 = st.columns(2)
        q_type = classify_question(question)

        # 3. 병렬 실행을 위한 헬퍼 함수 정의
        # 팁: 이제 run_rag는 vector_db 대신 검색된 docs를 직접 받도록 아래에서 수정할 겁니다.
        def fetch_gpt():
            try:
                if q_type == "calculation":
                    ans, _ = run_calculation_chain(question, "gpt", st.session_state.vector_db)
                else:
                    # run_rag에 shared_docs를 직접 넘겨줍니다.
                    ans = run_rag_final(question, answer_style, "gpt", gpt_h, shared_docs)
                
                refs = set([f"- {d.metadata['source']} p.{d.metadata['page'] + 1}" for d in shared_docs])
                return f"{ans}\n\n---\n**참고:**\n" + "\n".join(sorted(refs))
            except Exception as e:
                return f"⚠️ GPT 오류 발생: {str(e)}"

        def fetch_gemini():
            try:
                if q_type == "calculation":
                    ans, _ = run_calculation_chain(question, "gemini", st.session_state.vector_db)
                else:
                    ans = run_rag_final(question, answer_style, "gemini", gem_h, shared_docs)
                
                refs = set([f"- {d.metadata['source']} p.{d.metadata['page'] + 1}" for d in shared_docs])
                return f"{ans}\n\n---\n**참고:**\n" + "\n".join(sorted(refs))
            except Exception as e:
                return f"⚠️ Gemini 오류 발생: {str(e)}"

        # 4. 병렬 실행 시작
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_gpt = executor.submit(fetch_gpt)
            future_gemini = executor.submit(fetch_gemini)

            with col1:
                with st.chat_message("assistant", avatar="🤖"):
                    st.subheader("GPT-5.2")
                    p_gpt = st.empty()
                    p_gpt.info("GPT 분석 중...")
            
            with col2:
                with st.chat_message("assistant", avatar="♊"):
                    st.subheader("Gemini 2.5")
                    p_gem = st.empty()
                    p_gem.info("Gemini 분석 중...")

            # 결과 수집
            final_gpt = future_gpt.result()
            final_gemini = future_gemini.result()

            # 화면 업데이트 및 저장
            p_gpt.markdown(final_gpt)
            st.session_state.gpt_messages.append({"role": "assistant", "content": final_gpt})
            
            p_gem.markdown(final_gemini)
            st.session_state.gemini_messages.append({"role": "assistant", "content": final_gemini})