import hashlib
import time
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(
    page_title="Arnold's Cat Map 체험기",
    page_icon="🐈",
    layout="wide",
)


# -----------------------------
# Image preprocessing
# -----------------------------

def center_crop_to_square(img: Image.Image, size: int) -> Image.Image:
    """Crop the center square and resize to size x size."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def fit_with_padding_to_square(
    img: Image.Image,
    size: int,
    padding_color: tuple[int, int, int] = (20, 20, 24),
) -> Image.Image:
    """Fit the whole image into a square canvas, preserving aspect ratio."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), padding_color)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def make_demo_image(size: int) -> Image.Image:
    """Create a simple built-in demo image so the app works without upload."""
    img = Image.new("RGB", (size, size), (245, 247, 250))
    draw = ImageDraw.Draw(img)

    # grid
    step = max(16, size // 16)
    for x in range(0, size, step):
        draw.line((x, 0, x, size), fill=(220, 225, 232), width=1)
    for y in range(0, size, step):
        draw.line((0, y, size, y), fill=(220, 225, 232), width=1)

    # stylized cat face
    cx, cy = size // 2, size // 2
    r = size // 4
    face = (255, 210, 120)
    outline = (40, 45, 60)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=face, outline=outline, width=max(2, size // 128))

    # ears
    draw.polygon(
        [(cx - r * 3 // 4, cy - r * 3 // 4), (cx - r, cy - r * 7 // 4), (cx - r // 4, cy - r)],
        fill=face,
        outline=outline,
    )
    draw.polygon(
        [(cx + r * 3 // 4, cy - r * 3 // 4), (cx + r, cy - r * 7 // 4), (cx + r // 4, cy - r)],
        fill=face,
        outline=outline,
    )

    # eyes, nose, mouth
    eye_r = max(3, size // 40)
    draw.ellipse((cx - r // 2 - eye_r, cy - r // 5 - eye_r, cx - r // 2 + eye_r, cy - r // 5 + eye_r), fill=outline)
    draw.ellipse((cx + r // 2 - eye_r, cy - r // 5 - eye_r, cx + r // 2 + eye_r, cy - r // 5 + eye_r), fill=outline)
    draw.polygon([(cx, cy), (cx - size // 40, cy + size // 30), (cx + size // 40, cy + size // 30)], fill=(210, 80, 90))
    draw.arc((cx - size // 12, cy, cx, cy + size // 10), 10, 170, fill=outline, width=max(1, size // 160))
    draw.arc((cx, cy, cx + size // 12, cy + size // 10), 10, 170, fill=outline, width=max(1, size // 160))

    # whiskers
    for dy in [-size // 32, size // 64, size // 20]:
        draw.line((cx - r // 6, cy + dy, cx - r, cy + dy - size // 24), fill=outline, width=max(1, size // 170))
        draw.line((cx + r // 6, cy + dy, cx + r, cy + dy - size // 24), fill=outline, width=max(1, size // 170))

    # title text
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, size // 18))
        small_font = ImageFont.truetype("DejaVuSans.ttf", max(12, size // 32))
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    title = "Arnold's Cat Map"
    subtitle = "deterministic chaos"
    title_box = draw.textbbox((0, 0), title, font=font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=small_font)
    draw.text(((size - (title_box[2] - title_box[0])) // 2, size - size // 7), title, fill=outline, font=font)
    draw.text(((size - (subtitle_box[2] - subtitle_box[0])) // 2, size - size // 11), subtitle, fill=(90, 95, 110), font=small_font)

    return img


def image_to_png_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------
# Arnold's Cat Map
# -----------------------------

@st.cache_data(show_spinner=False)
def cat_maps(n: int):
    """Precompute coordinate maps for forward and inverse Arnold's Cat Map."""
    y, x = np.indices((n, n))

    # Forward map:
    # x' = x + y mod N
    # y' = x + 2y mod N
    forward_x = (x + y) % n
    forward_y = (x + 2 * y) % n

    # Inverse map of [[1,1],[1,2]] is [[2,-1],[-1,1]] modulo N.
    # Given current coordinate (x', y'), its previous coordinate is:
    # x = 2x' - y' mod N
    # y = -x' + y' mod N
    inverse_x = (2 * x - y) % n
    inverse_y = (-x + y) % n

    return y, x, forward_y, forward_x, inverse_y, inverse_x


def cat_forward(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    y, x, forward_y, forward_x, _, _ = cat_maps(n)
    out = np.empty_like(arr)
    out[forward_y, forward_x] = arr[y, x]
    return out


def cat_inverse(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    y, x, _, _, inverse_y, inverse_x = cat_maps(n)
    out = np.empty_like(arr)
    out[inverse_y, inverse_x] = arr[y, x]
    return out


def apply_steps(arr: np.ndarray, steps: int) -> np.ndarray:
    if steps >= 0:
        for _ in range(steps):
            arr = cat_forward(arr)
    else:
        for _ in range(-steps):
            arr = cat_inverse(arr)
    return arr


@st.cache_data(show_spinner=False)
def find_matrix_period(n: int, max_iter: int = 50_000) -> int | None:
    """Return the period of the cat-map matrix modulo n, if found."""
    a, b, c, d = 1, 1, 1, 2  # A
    ma, mb, mc, md = 1, 0, 0, 1  # I

    for k in range(1, max_iter + 1):
        # M <- A M mod n
        ma, mb, mc, md = (
            (a * ma + b * mc) % n,
            (a * mb + b * md) % n,
            (c * ma + d * mc) % n,
            (c * mb + d * md) % n,
        )
        if (ma, mb, mc, md) == (1, 0, 0, 1):
            return k
    return None


# -----------------------------
# Session state helpers
# -----------------------------

def init_state_from_image(img: Image.Image, source_key: str, period: int | None):
    st.session_state.source_key = source_key
    st.session_state.original_arr = np.array(img.convert("RGB"))
    st.session_state.current_arr = st.session_state.original_arr.copy()
    st.session_state.iteration = 0
    st.session_state.period = period


def current_pil_image() -> Image.Image:
    return Image.fromarray(st.session_state.current_arr.astype(np.uint8), mode="RGB")


def effective_iteration() -> int:
    period = st.session_state.get("period")
    it = st.session_state.get("iteration", 0)
    if period:
        return it % period
    return it


# -----------------------------
# UI
# -----------------------------

st.title("🐈 Arnold's Cat Map 체험기")
st.caption("이미지를 단순한 규칙으로 계속 섞으면, 무작위처럼 보이다가 다시 원래 모습으로 돌아옵니다.")

with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        help="업로드한 이미지는 Arnold's Cat Map 적용을 위해 정사각형으로 변환됩니다.",
    )

    size = st.selectbox(
        "내부 계산 크기",
        options=[256, 512, 768, 1024],
        index=1,
        help="512×512가 속도와 화질의 균형이 좋습니다.",
    )

    transform_mode_label = st.radio(
        "정사각형 변환 방식",
        options=["가운데 자르기", "전체 보존하기"],
        index=0,
        help="Arnold's Cat Map은 정사각형 격자에서 가장 깔끔하게 작동합니다.",
    )

    padding_label = "어두운 회색"
    if transform_mode_label == "전체 보존하기":
        padding_label = st.selectbox(
            "여백 색",
            options=["어두운 회색", "흰색", "검정", "연회색"],
            index=0,
        )

    padding_colors = {
        "어두운 회색": (20, 20, 24),
        "흰색": (255, 255, 255),
        "검정": (0, 0, 0),
        "연회색": (235, 238, 242),
    }
    padding_color = padding_colors[padding_label]

    st.divider()
    st.markdown("### 설명")
    st.markdown(
        """
        - **가운데 자르기**: 중앙을 기준으로 정사각형으로 잘라 사용합니다.
        - **전체 보존하기**: 원본 전체를 유지하고 남는 공간을 여백으로 채웁니다.
        """
    )

# Load and preprocess source image
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    raw_hash = hashlib.md5(raw_bytes).hexdigest()
    source_img = Image.open(BytesIO(raw_bytes))
else:
    raw_hash = "demo"
    source_img = make_demo_image(size)

mode_key = "crop" if transform_mode_label == "가운데 자르기" else "fit"
source_key = f"{raw_hash}|{size}|{mode_key}|{padding_label}"

if mode_key == "crop":
    prepared_img = center_crop_to_square(source_img, size)
else:
    prepared_img = fit_with_padding_to_square(source_img, size, padding_color)

period = find_matrix_period(size)

if "source_key" not in st.session_state or st.session_state.source_key != source_key:
    init_state_from_image(prepared_img, source_key, period)

# Controls
left, right = st.columns([1.2, 1])

with left:
    st.subheader("이미지")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("1번 섞기", use_container_width=True):
            st.session_state.current_arr = cat_forward(st.session_state.current_arr)
            st.session_state.iteration += 1
            st.rerun()

    with c2:
        if st.button("10번 섞기", use_container_width=True):
            st.session_state.current_arr = apply_steps(st.session_state.current_arr, 10)
            st.session_state.iteration += 10
            st.rerun()

    with c3:
        if st.button("1번 되돌리기", use_container_width=True):
            st.session_state.current_arr = cat_inverse(st.session_state.current_arr)
            st.session_state.iteration -= 1
            st.rerun()

    with c4:
        if st.button("초기화", use_container_width=True):
            st.session_state.current_arr = st.session_state.original_arr.copy()
            st.session_state.iteration = 0
            st.rerun()

    with c5:
        jump_to_period = st.button("원래대로", use_container_width=True, disabled=period is None)
        if jump_to_period and period is not None:
            remaining = (-effective_iteration()) % period
            st.session_state.current_arr = apply_steps(st.session_state.current_arr, remaining)
            st.session_state.iteration += remaining
            st.rerun()

    frames = st.slider("자동 재생 프레임 수", 5, 200, 40, 5)
    speed = st.slider("자동 재생 속도: 프레임 간 대기 시간(초)", 0.00, 0.30, 0.04, 0.01)

    image_placeholder = st.empty()

    if st.button("자동 재생 시작", type="primary", use_container_width=True):
        for _ in range(frames):
            st.session_state.current_arr = cat_forward(st.session_state.current_arr)
            st.session_state.iteration += 1
            image_placeholder.image(
                current_pil_image(),
                caption=f"현재 반복 횟수: {effective_iteration()}" + (f" / 주기 {period}" if period else ""),
                use_container_width=True,
            )
            time.sleep(speed)

    image_placeholder.image(
        current_pil_image(),
        caption=f"현재 반복 횟수: {effective_iteration()}" + (f" / 주기 {period}" if period else ""),
        use_container_width=True,
    )

    st.download_button(
        "현재 이미지 PNG로 저장",
        data=image_to_png_bytes(current_pil_image()),
        file_name=f"arnolds_cat_map_step_{effective_iteration()}.png",
        mime="image/png",
        use_container_width=True,
    )

with right:
    st.subheader("수학적 구조")
    st.latex(r"""
    \begin{pmatrix}x' \\ y'\end{pmatrix}
    =
    \begin{pmatrix}1 & 1 \\ 1 & 2\end{pmatrix}
    \begin{pmatrix}x \\ y\end{pmatrix}
    \pmod{N}
    """)

    st.markdown(
        f"""
        현재 내부 격자 크기는 **{size}×{size}**입니다.

        - $x'=(x+y)\mod N$
        - $y'=(x+2y)\mod N$
        - 행렬식은 $1$이므로 픽셀 정보가 사라지지 않습니다.
        - 유한한 픽셀 격자 위에서는 이 변환이 일종의 **순열(permutation)** 이 됩니다.
        """
    )

    if period is not None:
        st.success(f"이 격자 크기에서 모든 픽셀이 원래 자리로 돌아오는 주기: {period}회")
    else:
        st.warning("설정한 탐색 한도 안에서 주기를 찾지 못했습니다.")

    st.markdown(
        """
        ### 관찰 포인트
        1. 처음 몇 번은 이미지가 찢어지고 접히는 것처럼 보입니다.
        2. 중간에는 거의 노이즈처럼 보입니다.
        3. 하지만 정보가 사라진 것이 아니라 위치가 재배열된 것입니다.
        4. 충분히 반복하면 다시 원래 이미지가 나타납니다.
        """
    )

st.divider()
st.markdown(
    """
    #### 사용 팁
    - 인물 사진이나 사물 사진은 **가운데 자르기**가 보통 더 예쁩니다.
    - 캡처 화면이나 문서 이미지는 **전체 보존하기**가 더 안전합니다.
    - 512×512는 웹 체험용으로 속도와 화질의 균형이 좋습니다.
    """
)
