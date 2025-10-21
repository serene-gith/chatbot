import os
import time
import streamlit as st

# (선택) OpenAI 사용
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Streamlit Chatbot", page_icon="💬", layout="centered")

# --- Sidebar: 설정 ---
with st.sidebar:
    st.title("⚙️ 설정")
    st.caption("OpenAI 키가 없으면 데모 모드로 동작합니다.")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", value=os.getenv("OPENAI_API_KEY", ""))
    model_name = st.text_input("모델 이름", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="You are a helpful Korean AI assistant. 답변은 가능한 한 간결하고 정확하게, 한국어로 해주세요.",
        height=120,
    )
    use_stream = st.toggle("스트리밍 출력", value=True)

    if st.button("대화 초기화 🧹", type="secondary"):
        st.session_state.clear()
        st.rerun()

st.title("💬 Streamlit Chatbot")

# --- 세션 상태 초기화 ---
if "history" not in st.session_state:
    # history에는 user/assistant 메시지만 저장 (system은 매 호출 시 주입)
    st.session_state.history = []

# --- 메시지 렌더링 ---
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 데모(오프라인) 봇: 간단한 규칙/에코 ---
def demo_bot_reply(user_text: str) -> str:
    text = user_text.strip().lower()
    if any(g in text for g in ["안녕", "hello", "hi", "hey"]):
        return "안녕하세요! 저는 데모 챗봇이에요. 무엇을 도와드릴까요?"
    if "시간" in text or "time" in text:
        return f"지금 시간 기준으로는 대략 {time.strftime('%H:%M')} 입니다."
    if "도움" in text or "help" in text:
        return "예: “파이썬 리스트 정렬 예시 알려줘”, “여행 일정 짜줘” 같이 물어보세요!"
    # 기본 에코
    return f"들으신 내용: “{user_text}” — OpenAI 키가 없으면 데모 모드로 이렇게 응답해요."

# --- OpenAI로 답변 생성 ---
def llm_reply(user_text: str, api_key: str, model: str, temp: float, stream: bool) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai 패키지를 찾을 수 없습니다. `pip install openai` 후 다시 시도하세요.")
    if not api_key:
        raise RuntimeError("API 키가 없습니다.")

    client = OpenAI(api_key=api_key)

    # system + history 조합
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.history)
    messages.append({"role": "user", "content": user_text})

    if stream:
        # 스트리밍 출력
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            stream_resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                stream=True,
            )
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
                placeholder.markdown(full_text)
            return full_text
    else:
        # 비스트리밍
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temp
        )
        return resp.choices[0].message.content

# --- 입력창 ---
user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    # 사용자 메시지 화면 및 히스토리 반영
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 답변 생성
    try:
        if api_key:
            answer = llm_reply(user_input, api_key, model_name, temperature, use_stream)
        else:
            # 데모 모드
            with st.chat_message("assistant"):
                answer = demo_bot_reply(user_input)
                st.markdown(answer)
    except Exception as e:
        # 오류 메시지 출력
        answer = f"오류가 발생했습니다: {e}"
        with st.chat_message("assistant"):
            st.error(answer)

    # 히스토리에 assistant 메시지 저장 (스트리밍일 때도 결과로 덮어씁니다)
    st.session_state.history.append({"role": "assistant", "content": answer})
