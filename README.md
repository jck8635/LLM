# 🛡️ 유해 표현 / 피싱·광고 탐지 & LLM 설명 대시보드

KcBERT의 1차 텍스트 분류 성능과 Gemma2(Ollama) + RAG의 심층 검증을 결합하여, 문장의 유해성 및 피싱·광고 여부를 실시간으로 탐지하고 그 이유를 자연어로 설명해 주는 Streamlit 대시보드 애플리케이션입니다.

동영상 시연 영상 : https://youtu.be/txKLPJgzyGY
---

## 🚀 주요 기능
- **1차 텍스트 분류**: 미세조정(Fine-tuned)된 KcBERT 모델을 활용해 문장을 '정상', '유해 표현', '피싱·광고' 3가지 클래스로 분류
- **2차 LLM 심층 심사**: 1차 분류 결과가 '정상'이더라도 문맥 속 우회 표현(숫자/특수문자 낀 비속어 등)이 존재할 경우, Gemma2 모델이 최종 판정을 교정
- **RAG 기반 설명 생성**: Chroma 벡터 스토어에 저장된 유해성 판단 기준 문서를 검색(Retriever)하여 설득력 있는 분석 사유 제공
- **대시보드 시각화**: Plotly를 이용한 클래스별 예측 확률 바 차트 및 실시간 인터랙티브 UI 제공

---

## 🛠️ 기술 스택 (Tech Stack)
- **Frontend / UI**: Streamlit, Plotly
- **NLP / Model**: PyTorch, Hugging Face Transformers (`KcBERT`), Ollama (`Gemma2`)
- **RAG / Orchestration**: LangChain, ChromaDB, Hugging Face Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Preprocessing**: `soynlp`, `emoji`, `re`

---

## 📁 프로젝트 구조
```text
.
├── app.py                 # Streamlit 메인 애플리케이션 코드
├── train.ipynb            # 학습 영역
├── model/                 # KcBERT 파인튜닝 모델 폴더 (Git LFS 관리 권장)
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── chroma_db/             # Chroma RAG 벡터 DB (실행 시 자동 생성)
├── dataset/               # 학습 및 평가 데이터셋 폴더
├── requirements.txt
├── .gitattributes
├── .gitignore             # Git 제외 파일 설정
└── README.md


⚙️ 설치 및 실행 가이드
1. 사전 요구사항
Python 3.10 이상 권장

Ollama 설치 및 gemma2:latest 모델 다운로드 완료 상태

Bash
ollama run gemma2:latest
2. 레포지토리 클론 및 가상환경 설정
Bash
git clone [https://github.com/아이디/레포지토리이름.git](https://github.com/아이디/레포지토리이름.git)
cd 레포지토리이름

# 가상환경 생성 및 활성화
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
3. 패키지 설치
Bash
pip install streamlit torch pandas plotly transformers emoji soynlp langchain-ollama langchain-community langchain-core sentence-transformers chromadb
4. 모델 준비
model/ 폴더 안에 파인튜닝된 KcBERT 모델 파일들을 배치합니다. (대용량 파일 업로드 관련 내용은 아래 참고)

5. 애플리케이션 실행
Bash
streamlit run app.py
