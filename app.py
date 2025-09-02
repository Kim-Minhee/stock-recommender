# app.py
import datetime as dt
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------- 페이지 기본 설정 ----------
st.set_page_config(
    page_title="stock-recommender (base)",
    page_icon="📈",
    layout="wide",
)

# ---------- 사이드바 ----------
st.sidebar.title("⚙️ 설정")
default_ticker = "AAPL"
ticker = st.sidebar.text_input("종목 티커", value=default_ticker, help="예: AAPL, MSFT, TSLA, NVDA")
period = st.sidebar.selectbox("기간", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("간격", ["1d", "1wk", "1mo"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: 한국 종목은 `005930.KS`(삼성전자) 처럼 거래소 접미사를 붙이세요.")

# ---------- 헤더 ----------
st.title("📈 stock-recommender (기초 화면)")
st.caption("연습용 사이드 프로젝트 · Streamlit + yfinance")

# ---------- 데이터 로더 ----------
@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_price(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if not df.empty:
        df.index = pd.to_datetime(df.index)  # ensure datetime index
    return df

# ---------- 본문 레이아웃 ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"차트 · {ticker.upper()} · {period} / {interval}")

    if st.button("데이터 불러오기", type="primary"):
        df = load_price(ticker, period, interval)
        if df.empty:
            st.warning("데이터가 없습니다. 티커/기간을 확인해 주세요.")
        else:
            st.success(f"{ticker.upper()} 데이터 로드 완료")
            st.line_chart(df["Close"], height=320, use_container_width=True)

            # 간단 지표(기초)
            last_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
            chg = last_close - prev_close
            chg_pct = (chg / prev_close) * 100 if prev_close != 0 else 0.0

            m1 = df["Close"].tail(20).mean() if len(df) >= 20 else None
            m2 = df["Close"].tail(60).mean() if len(df) >= 60 else None

            k1, k2, k3 = st.columns(3)
            k1.metric("종가", f"{last_close:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
            k2.metric("20일 평균", f"{m1:,.2f}" if m1 else "데이터 부족")
            k3.metric("60일 평균", f"{m2:,.2f}" if m2 else "데이터 부족")

            with st.expander("원본 데이터 보기", expanded=False):
                st.dataframe(df.tail(200), use_container_width=True, height=260)

with col_right:
    st.subheader("추천(Placeholder)")
    st.info("여기는 나중에 **AI 추천 모델** 출력 영역이에요.\n\n예: 점수(0~1), 매수/보류/매도 제안, 근거 요약 등")
    st.markdown("---")

    def simple_signal(price: pd.Series) -> Optional[str]:
        if len(price) < 60:
            return None
        ma_short = price.rolling(20).mean()
        ma_long = price.rolling(60).mean()
        if ma_short.iloc[-1] > ma_long.iloc[-1]:
            return "⚡ 모멘텀 상향 (관심)"
        elif ma_short.iloc[-1] < ma_long.iloc[-1]:
            return "🧊 모멘텀 하향 (주의)"
        return "➖ 중립"

    # 미리보기용 샘플 계산 버튼
    if st.button("간단 모멘텀 신호 계산"):
        df = load_price(ticker, period, interval)
        if df.empty:
            st.warning("데이터가 없습니다.")
        else:
            sig = simple_signal(df["Close"])
            if sig is None:
                st.info("데이터가 부족해요. 기간을 늘려보세요.")
            else:
                st.success(f"{ticker.upper()} · {sig}")

# ---------- 푸터 ----------
st.markdown("---")
st.caption(
    f"Last update: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    "Roadmap: 데이터 소스 확장 → 피처 엔지니어링 → ML/LLM 추천 모델 연결 → 배포(Streamlit cloud)"
)