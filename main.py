import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_CLASS_YEAR = 2025
MIN_OBSERVATION_DAYS = 300

# 최근 20년
RECENT_START_YEAR = LAST_CLASS_YEAR - 19
RECENT_END_YEAR = LAST_CLASS_YEAR


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 데이터만 사용
    df = df[df["연도"] <= LAST_CLASS_YEAR]

    # 연도별 평균기온과 관측일수 계산
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    # 평균기온 결측값 제외
    annual = annual.dropna(subset=["연평균기온"])

    # 1908년부터 지난 연수
    annual["1908년부터_지난_연수"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# --------------------------------------------------
# 전체 기간 회귀분석
# --------------------------------------------------
x_all = annual["1908년부터_지난_연수"].to_numpy()
y_all = annual["연평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)

# 1년당 변화량 → 100년당 변화량
slope_all_100 = slope_all * 100

annual["전체_회귀예측"] = (
    slope_all * x_all + intercept_all
)


# --------------------------------------------------
# 최근 20년 회귀분석
# --------------------------------------------------
recent = annual[
    (annual["연도"] >= RECENT_START_YEAR)
    & (annual["연도"] <= RECENT_END_YEAR)
].copy()

x_recent = (
    recent["연도"] - BASE_YEAR
).to_numpy()

y_recent = recent["연평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)

# 100년당 변화량
slope_recent_100 = slope_recent * 100

recent["최근20년_회귀예측"] = (
    slope_recent * x_recent
    + intercept_recent
)


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
correlation = annual["연도"].corr(
    annual["연평균기온"]
)


# --------------------------------------------------
# 사용 기간
# --------------------------------------------------
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)

recent_start = int(recent["연도"].min())
recent_end = int(recent["연도"].max())
recent_count = len(recent)


# --------------------------------------------------
# 함수: 기울기 표시 문구
# --------------------------------------------------
def slope_text(value):
    if value > 0:
        return f"100년에 {value:.2f}℃ 상승"
    elif value < 0:
        return f"100년에 {abs(value):.2f}℃ 하락"
    else:
        return "100년에 변화 없음"


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용해 기온 변화 추세를 살펴보고, "
    "선형 회귀를 이용해 미래의 예상 평균기온을 계산합니다."
)


# --------------------------------------------------
# 회귀 기울기 비교
# --------------------------------------------------
st.subheader("🔥 기온 상승 속도 비교")

st.write(
    "회귀 직선의 기울기를 이해하기 쉽게 "
    "**100년 동안 기온이 몇 ℃ 변하는지**로 나타냈습니다."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        slope_text(slope_all_100),
        delta=f"{slope_all_100:+.2f} ℃ / 100년",
    )

    st.caption(
        f"{start_year}~{end_year}년, "
        f"{year_count}개 연도 사용"
    )

with col2:
    st.metric(
        "최근 20년",
        slope_text(slope_recent_100),
        delta=f"{slope_recent_100:+.2f} ℃ / 100년",
    )

    st.caption(
        f"{recent_start}~{recent_end}년, "
        f"{recent_count}개 연도 사용"
    )


# --------------------------------------------------
# 전체 데이터 정보
# --------------------------------------------------
st.subheader("📌 회귀 직선에 사용한 데이터")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "사용한 연도 수",
    f"{year_count}개"
)

col2.metric(
    "시작 연도",
    f"{start_year}년"
)

col3.metric(
    "끝 연도",
    f"{end_year}년"
)

col4.metric(
    "상관계수",
    f"{correlation:.3f}"
)

st.caption(
    f"2025년 이후 데이터와 연간 평균기온 관측일이 "
    f"{MIN_OBSERVATION_DAYS}일 미만인 해는 제외했습니다."
)


# --------------------------------------------------
# 산점도 + 회귀 직선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)


# --------------------------------------------------
# 전체 기간 회귀선
# --------------------------------------------------
line_years = np.arange(
    start_year,
    end_year + 1
)

line_x = line_years - BASE_YEAR

line_temperature_all = (
    slope_all * line_x
    + intercept_all
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperature_all,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "전체 기간 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


# --------------------------------------------------
# 최근 20년 회귀선
# --------------------------------------------------
recent_line_years = np.arange(
    recent_start,
    recent_end + 1
)

recent_line_x = recent_line_years - BASE_YEAR

recent_line_temperature = (
    slope_recent * recent_line_x
    + intercept_recent
)

fig.add_trace(
    go.Scatter(
        x=recent_line_years,
        y=recent_line_temperature,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="데이터",
)

# 가로축은 실제 연도 표시
fig.update_xaxes(
    tickformat="d",
    range=[
        start_year - 2,
        end_year + 2
    ],
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("🧮 회귀 분석 결과")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")

    st.write(
        f"**회귀식**  \n"
        f"예상 기온 = "
        f"{slope_all:.5f} × (연도 - {BASE_YEAR}) "
        f"{intercept_all:+.3f}"
    )

    st.write(
        f"**100년당 변화량: "
        f"{slope_all_100:+.2f} ℃**"
    )

with col2:
    st.markdown("### 최근 20년")

    st.write(
        f"**회귀식**  \n"
        f"예상 기온 = "
        f"{slope_recent:.5f} × (연도 - {BASE_YEAR}) "
        f"{intercept_recent:+.3f}"
    )

    st.write(
        f"**100년당 변화량: "
        f"{slope_recent_100:+.2f} ℃**"
    )


# --------------------------------------------------
# 연도 선택 및 예측
# --------------------------------------------------
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - BASE_YEAR

predicted_temperature = (
    slope_all * selected_x
    + intercept_all
)

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-top: 15px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px;">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 60px;
            font-weight: bold;
            margin-top: 5px;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if (
    selected_year < start_year
    or selected_year > end_year
):
    st.warning(
        f"{selected_year}년은 회귀 직선에 사용한 "
        f"관측 기간 ({start_year}~{end_year}년) 밖이므로 "
        "회귀 직선을 연장한 추정값입니다."
    )


st.caption(
    "예상 기온은 과거 서울 기온의 선형 추세를 "
    "단순히 연장한 값입니다. 실제 미래 기후를 "
    "정밀하게 예측하는 기후모형의 결과는 아닙니다."
)
