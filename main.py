import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 기본 설정
# ============================================================

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
LAST_YEAR = 2025
MIN_OBSERVATION_DAYS = 300

TEST_START_YEAR = 2006
TEST_END_YEAR = 2025

TRAIN_50_START = 1956
TRAIN_50_END = 2005

# 사용자가 요청한 100년 학습 범위
TRAIN_100_REQUESTED_START = 1906
TRAIN_100_END = 2005


# ============================================================
# 데이터 불러오기
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 날짜가 없는 행 제거
    df = df.dropna(
        subset=["날짜"]
    )

    # 연도 생성
    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일수 300일 미만인 연도 제거
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    # 평균기온이 없는 연도 제거
    annual = annual.dropna(
        subset=["연평균기온"]
    )

    # 1908년부터 지난 연수
    annual["지난연수"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# ============================================================
# 전체 데이터 회귀
# ============================================================

X_all = annual[["지난연수"]]
y_all = annual["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

annual["전체예측"] = model_all.predict(X_all)

all_slope = model_all.coef_[0]
all_slope_100 = all_slope * 100
all_intercept = model_all.intercept_

all_mae = mean_absolute_error(
    y_all,
    annual["전체예측"]
)

all_mse = mean_squared_error(
    y_all,
    annual["전체예측"]
)

all_r2 = r2_score(
    y_all,
    annual["전체예측"]
)


# ============================================================
# 훈련 / 테스트 데이터 분리
# ============================================================

train_50 = annual[
    (annual["연도"] >= TRAIN_50_START)
    & (annual["연도"] <= TRAIN_50_END)
].copy()

train_100 = annual[
    (annual["연도"] >= TRAIN_100_REQUESTED_START)
    & (annual["연도"] <= TRAIN_100_END)
].copy()

test = annual[
    (annual["연도"] >= TEST_START_YEAR)
    & (annual["연도"] <= TEST_END_YEAR)
].copy()


# ============================================================
# 최근 50년 모델
# ============================================================

X_train_50 = train_50[["지난연수"]]
y_train_50 = train_50["연평균기온"]

X_test = test[["지난연수"]]
y_test = test["연평균기온"]

model_50 = LinearRegression()

model_50.fit(
    X_train_50,
    y_train_50
)

pred_50 = model_50.predict(
    X_test
)

slope_50 = model_50.coef_[0]
slope_50_100 = slope_50 * 100

mae_50 = mean_absolute_error(
    y_test,
    pred_50
)

mse_50 = mean_squared_error(
    y_test,
    pred_50
)

r2_50 = r2_score(
    y_test,
    pred_50
)


# ============================================================
# 최근 100년 모델
# ============================================================

X_train_100 = train_100[["지난연수"]]
y_train_100 = train_100["연평균기온"]

model_100 = LinearRegression()

model_100.fit(
    X_train_100,
    y_train_100
)

pred_100 = model_100.predict(
    X_test
)

slope_100 = model_100.coef_[0]
slope_100_100 = slope_100 * 100

mae_100 = mean_absolute_error(
    y_test,
    pred_100
)

mse_100 = mean_squared_error(
    y_test,
    pred_100
)

r2_100 = r2_score(
    y_test,
    pred_100
)


# ============================================================
# 화면 제목
# ============================================================

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온을 이용해 선형회귀 모델을 만들고, "
    "과거 데이터를 학습한 모델이 최근 20년의 기온을 "
    "얼마나 잘 예측하는지 평가합니다."
)


# ============================================================
# 데이터 사용 범위
# ============================================================

st.subheader("📊 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{annual['연도'].min()}~{annual['연도'].max()}"
    )

with col2:
    st.metric(
        "테스트 데이터",
        f"{TEST_START_YEAR}~{TEST_END_YEAR}"
    )

with col3:
    st.metric(
        "테스트 연도 수",
        f"{len(test)}개"
    )

st.info(
    "관측일수가 300일 미만인 연도와 2025년 이후 데이터는 "
    "분석에서 제외했습니다. 원본 데이터는 1907년 10월부터 "
    "시작하므로 100년 학습 구간(1906~2005년)에서 실제로 "
    "사용 가능한 첫 연도는 1908년입니다."
)


# ============================================================
# 전체 데이터 회귀 평가
# ============================================================

st.subheader("📈 전체 데이터로 만든 회귀 모델")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용 연도 수",
        f"{len(annual)}개"
    )

with col2:
    st.metric(
        "100년당 변화",
        f"{all_slope_100:+.2f} ℃"
    )

with col3:
    st.metric(
        "MAE",
        f"{all_mae:.3f} ℃"
    )

with col4:
    st.metric(
        "R²",
        f"{all_r2:.3f}"
    )

st.write(
    f"전체 기간 회귀선의 기울기: "
    f"**100년에 {all_slope_100:+.2f}℃**"
)


# ============================================================
# 전체 데이터 회귀선 그래프
# ============================================================

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

line_years = np.arange(
    annual["연도"].min(),
    annual["연도"].max() + 1
)

line_x = line_years - BASE_YEAR

line_pred = (
    all_slope * line_x
    + all_intercept
)

fig_all.add_trace(
    go.Scatter(
        x=line_years,
        y=line_pred,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
)

fig_all.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_all,
    use_container_width=True
)


# ============================================================
# 50년 vs 100년 모델 비교
# ============================================================

st.subheader(
    "🔬 최근 50년 학습 vs 최근 100년 학습"
)

st.write(
    "두 모델 모두 **2006~2025년을 동일한 테스트 데이터**로 "
    "사용합니다. 따라서 MAE, MSE, R²를 직접 비교할 수 있습니다."
)


# ------------------------------------------------------------
# 학습 데이터 정보
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟦 최근 50년 모델")
    st.write(
        f"학습 기간: **{TRAIN_50_START}~{TRAIN_50_END}년**"
    )
    st.write(
        f"학습 연도 수: **{len(train_50)}개**"
    )

with col2:
    st.markdown("### 🟩 최근 100년 모델")
    st.write(
        f"요청 학습 기간: **{TRAIN_100_REQUESTED_START}~{TRAIN_100_END}년**"
    )
    st.write(
        f"실제 학습 기간: **{train_100['연도'].min()}~{TRAIN_100_END}년**"
    )
    st.write(
        f"학습 연도 수: **{len(train_100)}개**"
    )


# ============================================================
# 기울기 비교
# ============================================================

st.markdown("### 📐 회귀선 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 모델",
        f"{slope_50_100:+.2f} ℃ / 100년"
    )

with col2:
    st.metric(
        "최근 100년 모델",
        f"{slope_100_100:+.2f} ℃ / 100년"
    )


# ============================================================
# 성능 비교
# ============================================================

st.markdown("### 🎯 테스트 데이터 예측 성능")

comparison = pd.DataFrame(
    {
        "모델": [
            "최근 50년 학습",
            "최근 100년 학습",
        ],
        "학습 기간": [
            "1956~2005",
            f"{train_100['연도'].min()}~2005",
        ],
        "테스트 기간": [
            "2006~2025",
            "2006~2025",
        ],
        "MAE (℃)": [
            mae_50,
            mae_100,
        ],
        "MSE (℃²)": [
            mse_50,
            mse_100,
        ],
        "R²": [
            r2_50,
            r2_100,
        ],
    }
)

st.dataframe(
    comparison.style.format(
        {
            "MAE (℃)": "{:.3f}",
            "MSE (℃²)": "{:.3f}",
            "R²": "{:.3f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 성능을 크게 표시
# ============================================================

st.markdown("### 🏆 테스트 성능 한눈에 비교")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.caption("50년 MAE")
    st.metric(
        "",
        f"{mae_50:.3f}"
    )

with col2:
    st.caption("100년 MAE")
    st.metric(
        "",
        f"{mae_100:.3f}"
    )

with col3:
    st.caption("50년 MSE")
    st.metric(
        "",
        f"{mse_50:.3f}"
    )

with col4:
    st.caption("100년 MSE")
    st.metric(
        "",
        f"{mse_100:.3f}"
    )

with col5:
    st.caption("50년 R²")
    st.metric(
        "",
        f"{r2_50:.3f}"
    )

with col6:
    st.caption("100년 R²")
    st.metric(
        "",
        f"{r2_100:.3f}"
    )


# ============================================================
# 테스트 데이터 실제값 vs 예측값
# ============================================================

st.subheader(
    "🔮 2006~2025년 실제 기온과 예측 기온"
)

prediction_df = test[
    ["연도", "연평균기온"]
].copy()

prediction_df["최근 50년 예측"] = pred_50
prediction_df["최근 100년 예측"] = pred_100

fig_test = go.Figure()

# 실제값
fig_test.add_trace(
    go.Scatter(
        x=prediction_df["연도"],
        y=prediction_df["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(width=3),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=prediction_df["연도"],
        y=prediction_df["최근 50년 예측"],
        mode="lines",
        name="50년 학습 모델 예측",
        line=dict(
            width=2,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 100년 모델
fig_test.add_trace(
    go.Scatter(
        x=prediction_df["연도"],
        y=prediction_df["최근 100년 예측"],
        mode="lines",
        name="100년 학습 모델 예측",
        line=dict(
            width=2,
            dash="dot"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
)

fig_test.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# ============================================================
# 어떤 모델이 더 좋은가?
# ============================================================

st.subheader("💡 두 모델 비교 결과")

if mae_50 < mae_100:
    mae_result = (
        "MAE는 최근 50년 모델이 더 작아 "
        "평균적인 예측 오차가 더 작습니다."
    )
else:
    mae_result = (
        "MAE는 최근 100년 모델이 더 작아 "
        "평균적인 예측 오차가 더 작습니다."
    )

if mse_50 < mse_100:
    mse_result = (
        "MSE도 최근 50년 모델이 더 작습니다."
    )
else:
    mse_result = (
        "MSE도 최근 100년 모델이 더 작습니다."
    )

if r2_50 > r2_100:
    r2_result = (
        "R²는 최근 50년 모델이 더 높아 "
        "테스트 기간의 기온 변화를 더 잘 설명합니다."
    )
else:
    r2_result = (
        "R²는 최근 100년 모델이 더 높아 "
        "테스트 기간의 기온 변화를 더 잘 설명합니다."
    )

st.write(
    f"""
    - **기울기:** 최근 50년 모델은 100년에
      **{slope_50_100:+.2f}℃**, 최근 100년 모델은
      **{slope_100_100:+.2f}℃**의 변화 추세를 학습했습니다.
    - **MAE:** {mae_result}
    - **MSE:** {mse_result}
    - **R²:** {r2_result}
    """
)


# ============================================================
# 평가 지표 설명
# ============================================================

with st.expander("📖 MAE, MSE, R²가 무엇인가요?"):
    st.markdown(
        """
        **MAE (평균 절대 오차)**  
        실제 기온과 예측 기온의 차이를 절댓값으로 바꾼 뒤 평균낸 값입니다.
        작을수록 좋습니다. 단위는 ℃입니다.

        **MSE (평균 제곱 오차)**  
        실제값과 예측값의 차이를 제곱한 뒤 평균낸 값입니다.
        큰 오차에 더 큰 벌점을 줍니다. 작을수록 좋습니다.

        **R² (결정계수)**  
        모델이 실제 기온의 변화를 얼마나 설명하는지를 나타냅니다.
        일반적으로 1에 가까울수록 좋습니다.

        이번 앱에서는 두 모델을 **완전히 같은 2006~2025년
        테스트 데이터**에 적용하기 때문에, 세 지표를 이용해
        어느 학습 기간이 더 잘 일반화되는지 비교할 수 있습니다.
        """
    )


# ============================================================
# 연도별 기온 예측
# ============================================================

st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - BASE_YEAR

prediction_all = model_all.predict(
    [[selected_x]]
)[0]

prediction_50 = model_50.predict(
    [[selected_x]]
)[0]

prediction_100 = model_100.predict(
    [[selected_x]]
)[0]


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터 모델",
        f"{prediction_all:.2f} ℃"
    )

with col2:
    st.metric(
        "50년 학습 모델",
        f"{prediction_50:.2f} ℃"
    )

with col3:
    st.metric(
        "100년 학습 모델",
        f"{prediction_100:.2f} ℃"
    )

if (
    selected_year < annual["연도"].min()
    or selected_year > annual["연도"].max()
):
    st.warning(
        f"{selected_year}년은 관측 데이터 기간 "
        f"({annual['연도'].min()}~{annual['연도'].max()}년) "
        "밖이므로 회귀선을 연장한 추정값입니다."
    )

st.caption(
    "회귀모델의 미래 예측값은 과거의 선형 추세를 "
    "연장한 통계적 추정값입니다."
)
