import json
import re
from pathlib import Path
import os

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import emoji
from soynlp.normalizer import repeat_normalize

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="유해·피싱 탐지 & LLM 설명 대시보드",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 유해 표현 / 피싱·광고 탐지 & LLM 설명 대시보드")
st.caption("KcBERT 분류 + Gemma-2-9B-IT (HF API 심층 검토) + LangChain + Chroma RAG")

# Streamlit Secrets에서 허깅페이스 토큰 안전하게 가져오기
if "HF_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]
else:
    st.error("Streamlit Secrets에 `HF_TOKEN`이 설정되어 있지 않습니다.")
    st.stop()

# =========================================================
# 2. 경로 설정
# =========================================================
MODEL_PATH = Path("model")  # 로컬 KcBERT 폴더
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {0: "정상", 1: "유해 표현", 2: "피싱·광고"}


# =========================================================
# 3. 전처리 함수
# =========================================================
FULL2HALF = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～ ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

emojis = "".join(emoji.EMOJI_DATA.keys())
pattern = re.compile(f"[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+")
url_pattern = re.compile(
    r"https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.translate(FULL2HALF)
    text = url_pattern.sub(" [URL] ", text)
    text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = repeat_normalize(text, num_repeats=2)
    return text


# =========================================================
# 4. 리소스 로드
# =========================================================
@st.cache_resource
def load_kcbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_rag_and_llm():
    rag_documents = [
        Document(
            page_content="욕설, 비하, 차별, 성적 모욕, 혐오 발언이 포함된 경우 유해 표현으로 분류한다.",
            metadata={"label": "유해 표현"}
        ),
        Document(
            page_content="특정 개인이나 집단을 비난하거나 조롱하는 표현은 유해 표현에 해당한다.",
            metadata={"label": "유해 표현"}
        ),
        Document(
            page_content="무료 제공, 클릭 유도, 짧은 링크, 계좌 이체 요구, 개인정보 입력 유도 등은 피싱·광고의 전형적인 패턴이다.",
            metadata={"label": "피싱·광고"}
        ),
        Document(
            page_content="의심스러운 URL, 비트링크, '지금 바로', '한정 특가', '당첨' 등의 문구가 있으면 피싱·광고 가능성이 높다.",
            metadata={"label": "피싱·광고"}
        ),
        Document(
            page_content="일상적인 대화, 질문, 정보 공유, 감정 표현 등은 정상으로 분류한다.",
            metadata={"label": "정상"}
        ),
        Document(
            page_content="단순한 비속어 하나가 있어도 전체 맥락이 공격적이지 않으면 정상일 수 있다.",
            metadata={"label": "정상"}
        ),
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=rag_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 허깅페이스 Inference API를 통한 Gemma-2-9B-IT 연동
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="google/gemma-2-9b-it",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=hf_endpoint)

    prompt = ChatPromptTemplate.from_template(
        """당신은 유해 콘텐츠와 피싱/광고를 면밀히 심사하는 보안 AI 전문가입니다.

[KcBERT 1차 분류 결과]
- 예측 라벨: {label}
- 신뢰도: {confidence:.1%}

[원문]
{text}

[참고할 수 있는 관련 지식]
{context}

[작업 지시사항]
1. 원문에 숫자, 특수문자, 초성, 은어 등을 섞어서 비속어나 욕설, 성적 모욕을 우회해서 표현한 경우가 있는지 면밀히 검토하세요.
2. 원문을 세심하게 분석하여 비속어, 비하, 욕설, 차별적 표현, 또는 의심스러운 피싱·광고 패턴이 포함되어 있는지 직접 검토하세요.
3. KcBERT가 '정상'으로 판단했더라도, 원문에 숨겨진 유해 표현이나 피싱 징후가 명백하다면 최종 판정을 '유해 표현' 또는 '피싱·광고'로 변경할 수 있습니다.
4. 반드시 아래의 JSON 형식에 맞춰서만 답변해주세요 (마크다운 코드블록 ```json ... ``` 사용 필수):

{{
  "final_label": "정상 또는 유해 표현 또는 피싱·광고 중 택 1",
  "explanation": "왜 이런 최종 판정을 내렸는지 구체적인 근거를 포함하여 3~5문장 내외로 작성"
}}
"""
    )

    chain = prompt | llm
    return retriever, chain


try:
    tokenizer, model = load_kcbert()
    retriever, explanation_chain = load_rag_and_llm()
except Exception as e:
    st.error(f"모델 또는 RAG/LLM을 불러오는 중 오류가 발생했습니다.\n\n{e}")
    st.stop()


# =========================================================
# 5. 분류 + LLM 심층 판단 함수
# =========================================================
def classify(text: str):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])

    return {
        "label_id": pred_id,
        "label": id2label[pred_id],
        "confidence": confidence,
        "probs": probs,
        "cleaned_text": cleaned
    }


def generate_llm_judgment(text: str, label: str, confidence: float):
    docs = retriever.invoke(text)
    context = "\n".join([f"- {doc.page_content}" for doc in docs])

    response = explanation_chain.invoke({
        "text": text,
        "label": label,
        "confidence": confidence,
        "context": context
    })
    
    content = response.content.strip()
    
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(content)
        return parsed.get("final_label", label), parsed.get("explanation", content)
    except Exception:
        return label, content


def classify_and_explain(text: str):
    kc_result = classify(text)
    final_label, explanation = generate_llm_judgment(
        text=text,
        label=kc_result["label"],
        confidence=kc_result["confidence"]
    )
    kc_result["final_label"] = final_label
    kc_result["explanation"] = explanation
    return kc_result


# =========================================================
# 6. 탭 구성
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔍 실시간 탐지 + LLM 심층 판단", "📊 모델 정보", "💡 사용 가이드"])

with tab1:
    st.subheader("문장 실시간 분석 + LLM 최종 심사")

    user_text = st.text_area(
        "분석할 문장을 입력하세요.",
        placeholder="예: 은근히 약올리네? 이거 신고해야 하나...",
        height=130
    )

    if st.button("분석 실행", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("문장을 입력해주세요.")
        else:
            with st.spinner("KcBERT 1차 분류 중... + 허깅페이스 Gemma-2-9B-IT가 심층 검토 중입니다..."):
                result = classify_and_explain(user_text)

            st.divider()

            final_label = result["final_label"]
            if final_label == "정상":
                st.success(f"✅ **최종 판정: 정상** (KcBERT 1차 예측: {result['label']} / 신뢰도: {result['confidence']*100:.1f}%)")
            else:
                st.error(f"🚨 **최종 판정: {final_label}** (KcBERT 1차 예측: {result['label']} / 신뢰도: {result['confidence']*100:.1f}%)")

            c1, c2, c3 = st.columns(3)
            c1.metric("정상 확률 (KcBERT)", f"{result['probs'][0]*100:.1f}%")
            c2.metric("유해 표현 확률 (KcBERT)", f"{result['probs'][1]*100:.1f}%")
            c3.metric("피싱·광고 확률 (KcBERT)", f"{result['probs'][2]*100:.1f}%")

            prob_df = pd.DataFrame({
                "클래스": ["정상", "유해 표현", "피싱·광고"],
                "확률": result["probs"]
            })
            fig = px.bar(
                prob_df, x="클래스", y="확률", text_auto=".1%",
                title="KcBERT 모델별 클래스 예측 확률", color="클래스",
                color_discrete_sequence=["#00CC96", "#EF553B", "#FFA15A"]
            )
            fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🤖 LLM 심층 심사 및 설명 (Gemma-2-9B-IT + RAG)")
            st.info(result["explanation"])

with tab2:
    st.subheader("사용 모델 정보")
    st.markdown("""
    | 구분 | 모델 | 역할 |
    |------|------|------|
    | 1차 분류 | **KcBERT** (fine-tuned) | 정상 / 유해 표현 / 피싱·광고 기본 확률 산출 |
    | 2차 심사 및 설명 | **Gemma-2-9B-IT** (Hugging Face API) | 1차 결과를 검토하여 최종 판정 수정 및 자연어 사유 생성 |
    | RAG | **Chroma + multilingual MiniLM** | 유해성 판단 기준 및 패턴 문서 검색 |
    | 오케스트레이션 | **LangChain** | 프롬프트 + LLM + Retriever 연결 |
    """)

with tab3:
    st.subheader("사용 및 실행 가이드")
    st.markdown("""
    - 허깅페이스 API를 통해 원격으로 `google/gemma-2-9b-it` 모델을 호출합니다.
    - Streamlit Cloud 환경에서도 메모리 부하 없이 원활하게 동작합니다.
    """)
