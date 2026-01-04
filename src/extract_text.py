import fitz  # pymupdf
import io
import base64
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def extract_documents_from_pdf(file_path: str, source_name: str):
    """
    모든 PDF 페이지를 이미지로 변환하여 Gemini Flash로 OCR을 수행합니다.
    텍스트, 표, 그림 설명을 모두 포함하여 상세한 문맥을 생성합니다.
    """
    # 모델명은 1.5-flash가 이미지 처리에 가장 효율적입니다.
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    doc = fitz.open(file_path)
    documents = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        
        # 1. 페이지를 고해상도 이미지로 변환
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        
        # 2. 이미지를 Base64로 인코딩
        encoded_image = base64.b64encode(img_data).decode("utf-8")
        
        # 3. Gemini에게 상세 분석 요청
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
        }
        
        text_message = {
            "type": "text",
            "text": (
                "너는 대학 강의 자료를 분석하는 전문가야. 이 페이지의 내용을 아주 상세하게 텍스트로 복원해줘.\n"
                "1. 모든 글자를 누락 없이 추출할 것.\n"
                "2. 표(Table)는 마크다운 형식을 사용하여 구조를 유지할 것.\n"
                "3. 그림이나 그래프가 있다면 무엇을 설명하는지 구체적으로 기술할 것.\n"
                "4. 수식이 있다면 가급적 텍스트나 LaTeX 형식으로 표현할 것."
            )
        }
        
        try:
            # 시각 정보를 포함한 메시지 전송
            response = vision_model.invoke([HumanMessage(content=[text_message, image_message])])
            page_content = response.content
        except Exception as e:
            page_content = f"OCR 분석 중 에러 발생 (페이지 {page_number}): {str(e)}"

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": source_name,
                    "page": page_number
                }
            )
        )

    doc.close()
    return documents

def split_documents(documents):
    # OCR로 생성된 텍스트는 정보 밀도가 높으므로 chunk 크기를 넉넉하게 잡습니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)