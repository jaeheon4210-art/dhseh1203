import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("1908년 이후 서울의 기온 데이터를 기반으로 연평균 기온 추이를 분석하고 회귀 모델로 예상 기온을 측정합니다.")

# 1. 데이터 불러오기 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime으로 변환 후 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 필터링
    df_filtered = df[df["연도"] <= 2025]
    
    # 연도별 관측일수(count) 및 평균기온의 평균(mean) 계산
    yearly_summary = df_filtered.groupby("연도")["평균기온"].agg(
        count="count",
        mean_temp="mean"
    ).reset_index()
    
    # 관측일수가 300일 이상인 해만 선별
    valid_data = yearly_summary[yearly_summary["count"] >= 300].copy()
    
    return valid_data

data = load_and_preprocess_data()

# 메타데이터 계산
num_years = len(data)
start_year = int(data["연도"].min())
end_year = int(data["연도"].max())

# 2. 선형 회귀 분석
# 독립변수 X: 1908년부터 지난 연수
data["X"] = data["연도"] - 1908
X = data["X"]
Y = data["mean_temp"]

# 1차 선형 회귀 계수(기울기, 절편) 구하기
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr = np.corrcoef(data["연도"], Y)[0, 1]

# 3. 화면 메타데이터 및 상관계수 표시
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("회귀선 활용 연도 수", f"{num_years}개 해")
with col2:
    st.metric("시작 연도", f"{start_year}년")
with col3:
    st.metric("끝 연도", f"{end_year}년")
with col4:
    st.metric("상관계수 (r)", f"{corr:.4f}")

st.divider()

# 4. 연도 선택 슬라이더 (1900 ~ 2100) 및 예측 결과 대형 표시
st.subheader("🔮 연도별 예상 기온 측정")
selected_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2026)

# 예측 계산: 선택 연도 - 1908
elapsed_years = selected_year - 1908
predicted_temp = slope * elapsed_years + intercept

st.metric(
    label=f"🎯 {selected_year}년 예상 연평균 기온",
    value=f"{predicted_temp:.2f} °C"
)

# 5. Plotly 시각화 (산점도 + 회귀 직선)
fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["mean_temp"],
    mode="markers",
    name="실제 연평균 기온",
    marker=dict(color="#1f77b4", size=7)
))

# 회귀 직선 (1900년 ~ 2100년 전체 구간에 대해 표시)
line_years = np.arange(1900, 2101)
line_X = line_years - 1908
line_Y = slope * line_X + intercept

fig.add_trace(go.Scatter(
    x=line_years,
    y=line_Y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="#ff7f0e", width=2)
))

# 현재 슬라이더로 선택한 위치 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"{selected_year}년 예측점",
    marker=dict(color="#2ca02c", size=12, symbol="diamond")
))

fig.update_layout(
    title="서울 연도별 평균기온 추이 및 선형 회귀선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
