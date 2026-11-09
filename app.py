import json
import re
from pathlib import Path

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import emoji
from soynlp.normalizer import repeat_normalize

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
st.caption("KcBERT 분류 + Hugging Face InferenceClient (Qwen) + Chroma RAG")

# =========================================================
# 2. 경로 설정
# =========================================================
MODEL_PATH = Path("model")  # 로컬 모델 폴더 경로
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
# 4. 리소스 로드 (KcBERT + RAG + Hugging Face Client)
# =========================================================
@st.cache_resource
def load_kcbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_rag_and_client():
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
    
    # 서버리스 Inference API에서 완벽히 지원되는 모델로 변경
    hf_api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
    
    hf_client = InferenceClient(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",  # 또는 "meta-llama/Llama-3.2-3B-Instruct"
        api_key=hf_api_key
    )
    
    return retriever, hf_client


try:
    tokenizer, model = load_kcbert()
    retriever, hf_client = load_rag_and_client()
except Exception as e:
    st.error(f"모델 또는 RAG/클라이언트를 불러오는 중 오류가 발생했습니다.\n\n{e}")
    st.stop()


# =========================================================
# 5. 분류 및 LLM 심층 판단 함수
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

    prompt = f"""당신은 유해 콘텐츠와 피싱/광고를 면밀히 심사하는 보안 AI 전문가입니다.

[KcBERT 1차 분류 결과]
- 예측 라벨: {label}
- 신뢰도: {confidence:.1%}

[원문]
{text}

[참고할 수 있는 관련 지식]
{context}

[작업 지시사항]
1. 원문을 세심하게 분석하여 비속어, 비하, 욕설, 차별적 표현, 또는 의심스러운 피싱·광고 패턴이 포함되어 있는지 직접 검토하세요.
2. KcBERT가 '정상'으로 판단했더라도, 원문에 숨겨진 유해 표현이나 피싱 징후가 명백하다면 최종 판정을 '유해 표현' 또는 '피싱·광고'로 변경할 수 있습니다.
3. 반드시 아래의 JSON 형식에 맞춰서만 답변해주세요 (마크다운 코드블록 ```json ... ``` 사용):

{{
  "final_label": "정상 또는 유해 표현 또는 피싱·광고 중 택 1",
  "explanation": "왜 이런 최종 판정을 내렸는지 구체적인 근거를 포함하여 3~5문장 내외로 작성"
}}
"""

    try:
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(content)
        return parsed.get("final_label", label), parsed.get("explanation", content)
    except Exception as e:
        return label, f"LLM 심사 중 오류 발생: {str(e)}"


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
# 6. 탭 구성 (UI)
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔍 실시간 탐지 + LLM 심층 판단", "📊 모델 정보", "💡 사용 가이드"])

with tab1:
    st.subheader("문장 실시간 분석 + LLM 최종 심사")

    user_text = st.text_area(
        "분석할 문장을 입력하세요.",
        placeholder="예: 지금 바로 클릭해서 무료로 받아가세요!",
        height=130
    )

    if st.button("분석 실행", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("문장을 입력해주세요.")
        else:
            with st.spinner("KcBERT 분류 및 허깅페이스 API 심층 검토 중..."):
                result = classify_and_explain(user_text)

            st.divider()

            final_label = result["final_label"]
            if final_label == "정상":
                st.success(f"✅ **최종 판정: 정상** (KcBERT 1차 예측: {result['label']} / 신뢰도: {result['confidence']*100:.1f}%)")
            else:
                st.error(f"🚨 **최종 판정: {final_label}** (KcBERT 1차 예측: {result['label']} / 신뢰도: {result['confidence']*100:.1f}%)")

            c1, c2, c3 = st.columns(3)
            c1.metric("정상 확률", f"{result['probs'][0]*100:.1f}%")
            c2.metric("유해 표현 확률", f"{result['probs'][1]*100:.1f}%")
            c3.metric("피싱·광고 확률", f"{result['probs'][2]*100:.1f}%")

            prob_df = pd.DataFrame({
                "클래스": ["정상", "유해 표현", "피싱·광고"],
                "확률": result["probs"]
            })
            fig = px.bar(
                prob_df, x="클래스", y="확률", text_auto=".1%",
                title="KcBERT 클래스별 예측 확률", color="클래스",
                color_discrete_sequence=["#00CC96", "#EF553B", "#FFA15A"]
            )
            fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🤖 LLM 심층 심사 및 설명 (Hugging Face API + RAG)")
            st.info(result["explanation"])


with tab2:
    st.subheader("사용 모델 정보")
    st.markdown("""
    | 구분 | 모델 | 역할 |
    |------|------|------|
    | 1차 분류 | **KcBERT** | 정상 / 유해 표현 / 피싱·광고 기본 확률 산출 |
    | 2차 심사 및 설명 | **Qwen2.5-1.5B-Instruct** (Hugging Face API) | API 기반 심층 검토 및 사유 생성 |
    | RAG | **Chroma + multilingual MiniLM** | 관련 기준 문서 검색 |
    """)

with tab3:
    st.subheader("사용 가이드")
    st.markdown("Hugging Face 공식 `InferenceClient`를 통해 클라우드 환경에서도 에러 없이 LLM 심층 검토 기능이 작동합니다.")
