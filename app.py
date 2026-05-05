import hashlib
import time
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="Arnold's Cat Map",
    page_icon="🐈",
    layout="wide",
)

MATRIX_PRESETS = {
    "Standard Cat Map": np.array([[1, 1], [1, 2]], dtype=np.int64),
    "Classic Cat Map": np.array([[2, 1], [1, 1]], dtype=np.int64),
    "Strong Mixing": np.array([[1, 2], [1, 3]], dtype=np.int64),
}

MATRIX_DESCRIPTIONS = {
    "Standard Cat Map": "가장 널리 알려진 형태 중 하나입니다.",
    "Classic Cat Map": "문헌에서 자주 등장하는 고전적인 형태입니다.",
    "Strong Mixing": "trace가 더 커서 더 강하게 섞이는 느낌을 줍니다.",
}


# =========================
# 이미지 전처리
# =========================

def create_default_demo_image(size: int = 512) -> Image.Image:
    """
    외부 파일 없이 사용할 수 있는 간단한 데모 이미지.
    고양이 느낌의 도형 이미지를 직접 생성한다.
    """
    img = Image.new("RGB", (size, size), (245, 242, 235))
    draw = ImageDraw.Draw(img)

    # 배경 격자
    step = size // 16
    for i in range(0, size, step):
        draw.line((i, 0, i, size), fill=(225, 222, 215), width=1)
        draw.line((0, i, size, i), fill=(225, 222, 215), width=1)

    # 고양이 얼굴
    cx, cy = size // 2, size // 2
    r = int(size * 0.25)

    # 귀
    draw.polygon(
        [
            (cx - r, cy - r // 2),
            (cx - int(r * 0.75), cy - int(r * 1.45)),
            (cx - int(r * 0.25), cy - r),
        ],
        fill=(60, 60, 60),
    )
    draw.polygon(
        [
            (cx + r, cy - r // 2),
            (cx + int(r * 0.75), cy - int(r * 1.45)),
            (cx + int(r * 0.25), cy - r),
        ],
        fill=(60, 60, 60),
    )

    # 얼굴
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=(70, 70, 70),
        outline=(30, 30, 30),
        width=4,
    )

    # 눈
    eye_r = size // 35
    draw.ellipse(
        (
            cx - r // 2 - eye_r,
            cy - r // 5 - eye_r,
            cx - r // 2 + eye_r,
            cy - r // 5 + eye_r,
        ),
        fill=(245, 245, 245),
    )
    draw.ellipse(
        (
            cx + r // 2 - eye_r,
            cy - r // 5 - eye_r,
            cx + r // 2 + eye_r,
            cy - r // 5 + eye_r,
        ),
        fill=(245, 245, 245),
    )

    # 코와 입
    draw.polygon(
        [
            (cx - size // 45, cy + size // 30),
            (cx + size // 45, cy + size // 30),
            (cx, cy + size // 15),
        ],
        fill=(230, 120, 130),
    )
    draw.arc(
        (cx - size // 12, cy + size // 25, cx, cy + size // 7),
        start=10,
        end=170,
        fill=(230, 230, 230),
        width=2,
    )
    draw.arc(
        (cx, cy + size // 25, cx + size // 12, cy + size // 7),
        start=10,
        end=170,
        fill=(230, 230, 230),
        width=2,
    )

    # 수염
    for dy in [-size // 25, 0, size // 25]:
        draw.line(
            (cx - r // 4, cy + dy, cx - r - size // 8, cy + dy - size // 30),
            fill=(235, 235, 235),
            width=2,
        )
        draw.line(
            (cx + r // 4, cy + dy, cx + r + size // 8, cy + dy - size // 30),
            fill=(235, 235, 235),
            width=2,
        )

    # 방향성을 알아보기 위한 색 블록
    block = size // 9
    draw.rectangle((20, 20, 20 + block, 20 + block), fill=(180, 80, 80))
    draw.rectangle((size - 20 - block, 20, size - 20, 20 + block), fill=(80, 130, 190))
    draw.rectangle((20, size - 20 - block, 20 + block, size - 20), fill=(80, 160, 100))
    draw.rectangle(
        (size - 20 - block, size - 20 - block, size - 20, size - 20),
        fill=(180, 150, 60),
    )

    return img


def center_crop_and_resize(image: Image.Image, size: int) -> Image.Image:
    """
    이미지 중앙을 기준으로 정사각형으로 자른 뒤 size x size로 리사이즈.
    """
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def fit_with_padding_and_resize(image: Image.Image, size: int) -> Image.Image:
    """
    원본 전체를 유지하면서 정사각형 캔버스 안에 넣는다.
    남는 공간은 어두운 회색 여백으로 채운다.
    """
    image = image.convert("RGB")
    w, h = image.size

    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), (24, 24, 24))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))

    return canvas


def image_to_png_bytes(array: np.ndarray) -> bytes:
    """
    numpy image array를 PNG bytes로 변환.
    """
    img = Image.fromarray(array.astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# =========================
# Arnold's Cat Map 핵심 로직
# =========================

@st.cache_data(show_spinner=False)
def get_mapping_indices(size: int, matrix_tuple: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    주어진 행렬 A에 대해 모든 픽셀 좌표의 이동 위치를 미리 계산한다.

    x' = a x + b y mod N
    y' = c x + d y mod N
    """
    a, b, c, d = matrix_tuple
    y, x = np.indices((size, size), dtype=np.int64)

    new_x = (a * x + b * y) % size
    new_y = (c * x + d * y) % size

    return new_y, new_x


def apply_cat_map(array: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    이미지 배열에 Arnold's Cat Map을 1회 적용한다.
    """
    size = array.shape[0]
    matrix_tuple = tuple(int(v) for v in matrix.flatten())

    new_y, new_x = get_mapping_indices(size, matrix_tuple)

    result = np.empty_like(array)
    result[new_y, new_x] = array

    return result


def inverse_matrix_det_one(matrix: np.ndarray) -> np.ndarray:
    """
    det = 1인 2x2 정수 행렬의 정수 역행렬을 구한다.

    A = [[a,b],[c,d]]이면
    A^{-1} = [[d,-b],[-c,a]]
    """
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    det = int(a * d - b * c)

    if det != 1:
        raise ValueError("현재 코드는 det = 1인 행렬만 역변환을 지원합니다.")

    return np.array([[d, -b], [-c, a]], dtype=np.int64)


def apply_inverse_cat_map(array: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Arnold's Cat Map의 역변환을 1회 적용한다.
    """
    inv = inverse_matrix_det_one(matrix)
    return apply_cat_map(array, inv)


@st.cache_data(show_spinner=False)
def matrix_period(size: int, matrix_tuple: tuple[int, int, int, int], max_steps: int = 200_000) -> int | None:
    """
    mod N에서 A^k = I가 되는 최소 k를 찾는다.
    너무 오래 걸리는 경우 None을 반환한다.
    """
    a, b, c, d = matrix_tuple
    matrix = np.array([[a, b], [c, d]], dtype=np.int64) % size
    identity = np.eye(2, dtype=np.int64) % size

    power = identity.copy()

    for k in range(1, max_steps + 1):
        power = (power @ matrix) % size
        if np.array_equal(power, identity):
            return k

    return None


def get_matrix_latex(matrix: np.ndarray) -> str:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    return rf"""
    A=
    \begin{{pmatrix}}
    {a} & {b}\\
    {c} & {d}
    \end{{pmatrix}}
    """


# =========================
# Session State 초기화
# =========================

if "base_array" not in st.session_state:
    st.session_state.base_array = None

if "current_array" not in st.session_state:
    st.session_state.current_array = None

if "iteration" not in st.session_state:
    st.session_state.iteration = 0

if "config_signature" not in st.session_state:
    st.session_state.config_signature = None


# =========================
# UI
# =========================

st.title("🐈 Arnold's Cat Map 체험기")

st.markdown(
    """
이미지를 정해진 선형변환으로 계속 섞어보세요.  
무작위처럼 보이지만, 사실은 **결정론적이고 가역적인 변환**입니다.
"""
)

st.caption("기획 및 출처: 김사무 · 취미로 배우는 수학")

left, right = st.columns([0.32, 0.68], gap="large")

with left:
    st.subheader("설정")

    uploaded_file = st.file_uploader(
        "이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
    )

    fit_mode = st.radio(
        "정사각형 변환 방식",
        options=["가운데 자르기", "전체 보존하기"],
        index=0,
        help="Arnold's Cat Map은 정사각형 격자 위에서 가장 깔끔하게 작동합니다.",
    )

    size = st.selectbox(
        "내부 계산 크기",
        options=[256, 512, 768, 1024],
        index=1,
        help="크기가 클수록 화질은 좋아지지만 계산이 느려질 수 있습니다.",
    )

    matrix_name = st.selectbox(
        "선형변환 선택",
        options=list(MATRIX_PRESETS.keys()),
        index=0,
    )

    matrix = MATRIX_PRESETS[matrix_name]
    matrix_tuple = tuple(int(v) for v in matrix.flatten())

    st.markdown(f"**{matrix_name}**")
    st.caption(MATRIX_DESCRIPTIONS[matrix_name])
    st.latex(get_matrix_latex(matrix))

    det = int(round(np.linalg.det(matrix)))
    trace = int(matrix[0, 0] + matrix[1, 1])

    st.markdown(
        f"""
- determinant: `{det}`
- trace: `{trace}`
"""
    )

    period = matrix_period(size, matrix_tuple)

    if period is None:
        st.info("이 설정의 주기를 제한 시간 안에 찾지 못했습니다.")
    else:
        st.success(f"이 설정에서는 {period}번 반복하면 원래 이미지로 돌아옵니다.")

    st.divider()

    st.subheader("조작")

    col_a, col_b = st.columns(2)
    with col_a:
        step_one = st.button("1번 섞기", use_container_width=True)
    with col_b:
        step_ten = st.button("10번 섞기", use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        undo_one = st.button("1번 되돌리기", use_container_width=True)
    with col_d:
        reset = st.button("초기화", use_container_width=True)

    restore = st.button("원래대로 돌아가기", use_container_width=True)

    st.divider()

    auto_play = st.checkbox("자동 재생")
    auto_delay = st.slider(
        "자동 재생 간격",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )


# =========================
# 이미지 로딩 및 전처리
# =========================

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    source_image = Image.open(BytesIO(uploaded_bytes))
else:
    image_hash = "default-demo-image"
    source_image = create_default_demo_image(512)

config_signature = f"{image_hash}-{fit_mode}-{size}-{matrix_name}"

if config_signature != st.session_state.config_signature:
    if fit_mode == "가운데 자르기":
        processed = center_crop_and_resize(source_image, size)
    else:
        processed = fit_with_padding_and_resize(source_image, size)

    base_array = np.array(processed, dtype=np.uint8)

    st.session_state.base_array = base_array
    st.session_state.current_array = base_array.copy()
    st.session_state.iteration = 0
    st.session_state.config_signature = config_signature


# =========================
# 버튼 동작 처리
# =========================

if step_one:
    st.session_state.current_array = apply_cat_map(st.session_state.current_array, matrix)
    st.session_state.iteration += 1

if step_ten:
    for _ in range(10):
        st.session_state.current_array = apply_cat_map(st.session_state.current_array, matrix)
    st.session_state.iteration += 10

if undo_one:
    st.session_state.current_array = apply_inverse_cat_map(st.session_state.current_array, matrix)
    st.session_state.iteration -= 1

if reset or restore:
    st.session_state.current_array = st.session_state.base_array.copy()
    st.session_state.iteration = 0


# =========================
# 결과 표시
# =========================

with right:
    st.subheader("결과")

    current_array = st.session_state.current_array
    current_iteration = st.session_state.iteration

    if period is not None:
        effective_iteration = current_iteration % period
        st.markdown(
            f"""
현재 반복 횟수: **{current_iteration}회**  
주기 기준 위치: **{effective_iteration} / {period}**
"""
        )
    else:
        st.markdown(f"현재 반복 횟수: **{current_iteration}회**")

    st.image(
        current_array,
        caption=f"{matrix_name} · {fit_mode} · {size}×{size}",
        use_container_width=True,
    )

    png_bytes = image_to_png_bytes(current_array)

    st.download_button(
        label="현재 이미지 PNG 다운로드",
        data=png_bytes,
        file_name="arnolds_cat_map_result.png",
        mime="image/png",
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown("### 무엇을 보고 있는 걸까?")

    st.markdown("각 픽셀의 좌표를 다음과 같은 방식으로 이동시킵니다.")

    st.latex(
        r"""
        \begin{pmatrix}
        x'\\
        y'
        \end{pmatrix}
        =
        A
        \begin{pmatrix}
        x\\
        y
        \end{pmatrix}
        \pmod{N}
        """
    )

    st.markdown(
        """
여기서 N은 이미지의 한 변의 픽셀 수입니다.  
행렬식이 1이므로 픽셀 정보는 사라지지 않고, 단지 위치가 재배열됩니다.

즉, 이미지는 엉망으로 섞이는 것처럼 보이지만 실제로는 정보가 보존됩니다.  
유한한 픽셀 격자 위에서는 결국 같은 상태가 반복되므로, 충분히 반복하면 원래 이미지가 다시 나타납니다.
"""
    )

    st.caption("기획 및 출처: 김사무 · 취미로 배우는 수학")


# =========================
# 자동 재생
# =========================

if auto_play:
    time.sleep(auto_delay)
    st.session_state.current_array = apply_cat_map(st.session_state.current_array, matrix)
    st.session_state.iteration += 1
    st.rerun()
