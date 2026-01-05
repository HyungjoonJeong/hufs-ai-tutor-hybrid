import streamlit as st
import os
import tempfile
import queue
import time
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
        llm = ChatGoogleGenerativeAI(model="gemini-2.5", temperature=0)

    # 2. 관련 문서 검색
# 함수 내부에서 st.session_state.vector_db 대신 vector_db 사용!
    docs = vector_db.similarity_search(question, k=7)
    context = "\n\n".join([d.page_content for d in docs])

    template = """
너는 대학 과목 계산 문제를 푸는 조교이다. 제공된 [문맥]의 공식과 수치를 바탕으로 문제를 풀어라.

[수식 작성 규칙 - 중요]
1. 모든 수학 공식이나 변수는 반드시 LaTeX 형식을 사용하라.
2. 문장 안의 짧은 수식은 $기호 하나로 감싸라. (예: $P(Z < z)$)
3. 별도의 줄에 표시해야 하는 복잡한 공식은 $$기호 두 개로 감싸라.
(예: $$f(x) = \\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}} \exp...$$)
4. 절대 [ ] 나 ( ) 로 수식을 감싸지 마라. 오직 $와 $$만 사용한다.

[표 작성 규칙 - 중요]
1. 데이터를 비교하거나 분포를 나타낼 때는 반드시 마크다운 표(Table) 형식을 사용하라.
2. 표의 헤더와 데이터 사이에는 반드시 구분선(---)을 넣어라.
3. 첫 번째 열의 제목에 특수기호(|)를 단독으로 쓰지 말고, '구분' 또는 'Y \ X' 처럼 텍스트로 명확히 적어라.
   (예: | Y \ X | 0 | 1 | 2 |)
4. 중간에 설명을 넣기 위해 표를 끊지 말고, 표를 먼저 완벽히 완성한 후 설명을 덧붙여라.

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
    prompt = template.format(
        length_instruction=length_instruction,
        chat_history=chat_history_str,
        context=context_text,
        question=question
    )

    # 3. 스트리밍 루프
    try:
        for chunk in llm.stream(prompt):
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)
            
            if content:
                yield content
    except Exception as e:
        yield f"\n⚠️ 모델({model_type}) 호출 에러: {str(e)}"


# --------------------------------
# 일반 RAG 체인
# --------------------------------
# 2. 메인 답변 체인 (GPT-4o 사용 - 정밀한 논리)
# run_rag 정의 부분 수정
def run_rag_stream(question: str, answer_style: str, model_type: str, chat_history: list, docs: list):
    try:
        # 1. 모델 설정 (최신 모델명 반영)
        if model_type == "gpt":
            llm = ChatOpenAI(model="gpt-5.2", temperature=0.7, streaming=True)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5", temperature=0.7, streaming=True)

        # 2. 컨텍스트 및 히스토리 구성
        context_text = "\n\n".join([d.page_content for d in docs])
        chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in (chat_history or [])])
        
        length_instruction = (
            "핵심 위주로 번호를 매겨 간결하게 답하라." if answer_style == "짧게" 
            else "상세한 설명과 함께 단계별로 번호를 매겨 자세히 답하라."
        )

        template = """

당신은 한국외국어대학교의 1타 강사 AI 튜터입니다. 
제공된 [강의 자료]를 바탕으로 학생의 질문에 답변하세요.

[수식 작성 규칙 - 중요]
1. 모든 수학 공식이나 변수는 반드시 LaTeX 형식을 사용하라.
2. 문장 안의 짧은 수식은 $기호 하나로 감싸라. (예: $P(Z < z)$)
3. 별도의 줄에 표시해야 하는 복잡한 공식은 $$기호 두 개로 감싸라.
(예: $$f(x) = \\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}} \exp...$$)
4. 절대 [ ] 나 ( ) 로 수식을 감싸지 마라. 오직 $와 $$만 사용한다.

[표 작성 규칙 - 중요]
1. 데이터를 비교하거나 분포를 나타낼 때는 반드시 마크다운 표(Table) 형식을 사용하라.
2. 표의 헤더와 데이터 사이에는 반드시 구분선(---)을 넣어라.
3. 첫 번째 열의 제목에 특수기호(|)를 단독으로 쓰지 말고, '구분' 또는 'Y \ X' 처럼 텍스트로 명확히 적어라.
   (예: | Y \ X | 0 | 1 | 2 |)
4. 중간에 설명을 넣기 위해 표를 끊지 말고, 표를 먼저 완벽히 완성한 후 설명을 덧붙여라.

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

[문맥]
{context}

[질문]
{question}

답변:"""
        # .format 인자 정확히 매칭
        formatted_prompt = template.format(
            context=context_text,
            question=question
        )

        # 3. 스트리밍 루프
        for chunk in llm.stream(formatted_prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content

    except Exception as e:
        # 에러 발생 시 yield로 에러 메시지를 화면에 출력
        yield f"\n\n❌ [모델 에러] {model_type}: {str(e)}"



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
        # 1. 공통 검색
        with st.spinner("자료 찾는 중..."):
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
            shared_docs = retriever.invoke(question)

        # 2. 메시지 기록
        st.session_state.gpt_messages.append({"role": "user", "content": question})
        st.session_state.gemini_messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)

        # 3. 화면 공간 확보
        col1, col2 = st.columns(2)
        with col1:
            st.info("🤖 GPT-5.2")
            area_gpt = st.empty()
        with col2:
            st.info("♊ Gemini 2.5")
            area_gem = st.empty()

        # 4. 생성기 생성
        gen_gpt = run_rag_stream(question, answer_style, "gpt", st.session_state.gpt_messages[:-1], shared_docs)
        gen_gem = run_rag_stream(question, answer_style, "gemini", st.session_state.gemini_messages[:-1], shared_docs)

        # 5. 스트리밍 실행 (Queue 방식)
        q_gpt, q_gem = queue.Queue(), queue.Queue()
        
        def produce(gen, q):
            for chunk in gen:
                q.put(chunk)
            q.put(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(produce, gen_gpt, q_gpt)
            executor.submit(produce, gen_gem, q_gem)

            full_gpt, full_gem = "", ""
            gpt_done, gem_done = False, False

            while not (gpt_done and gem_done):
                # GPT 데이터 업데이트
                while not q_gpt.empty():
                    item = q_gpt.get()
                    if item is None: gpt_done = True
                    else: 
                        full_gpt += item
                        area_gpt.markdown(full_gpt + "▌")
                
                # Gemini 데이터 업데이트
                while not q_gem.empty():
                    item = q_gem.get()
                    if item is None: gem_done = True
                    else: 
                        full_gem += item
                        area_gem.markdown(full_gem + "▌")
                
                time.sleep(0.05)

        # 6. 마무리 저장
        area_gpt.markdown(full_gpt)
        area_gem.markdown(full_gem)
        st.session_state.gpt_messages.append({"role": "assistant", "content": full_gpt})
        st.session_state.gemini_messages.append({"role": "assistant", "content": full_gem})