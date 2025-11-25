# app.py
import streamlit as st
from PIL import Image
import io
import os
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold

# 1. 设置 Streamlit 页面配置
st.set_page_config(
    page_title="Gemini 智能数据提取器 (Demo)",
    layout="wide",
)

# 2. 检查 API Key 是否存在于 Streamlit Secrets 中
# Streamlit Secrets 是一个安全的存储环境变量的地方
try:
    # 尝试从 Streamlit Secrets 中读取 API Key
    # 部署时，这个环境变量将由 Streamlit Cloud 提供
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # 如果本地运行，或者 Secrets 未设置，则让用户输入
    st.error("⚠️ 错误：API 密钥未配置。")
    st.info("请点击左侧菜单栏的 'Settings'，输入你的 Gemini API Key，或在部署到 Streamlit Cloud 时配置 Secrets。")
    st.stop()

# 3. 初始化 Gemini 客户端
# 配置安全设置，避免模型因为内容略微敏感而拒绝输出
client = genai.Client(api_key=API_KEY)

# 将安全配置提取为一个单独的变量
# 注意：HarmCategory 和 HarmBlockThreshold 必须是 types.SafetySetting 对象
SAFETY_CONFIG = [
    genai.types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    )
    # 你可以在这里添加其他安全设置
]
# 使用 1.5 Flash 模型，它具有多模态和长上下文能力，且推理速度快、成本低
MODEL = "gemini-1.5-flash"


# 4. 核心逻辑函数：调用 Gemini API 进行图片分析
@st.cache_data(show_spinner="⚙️ 正在调用 Gemini 1.5 Flash 分析图片并提取数据...")
def extract_data_from_image(image_bytes, prompt_text):
    """
    接收图片字节流和提示词，调用 Gemini API，并返回文本结果。
    """
    try:
        # 将图片字节流转化为 API 要求的 Part 对象
        image_part = genai.types.Part.from_bytes(
            data=image_bytes.getvalue(),
            mime_type='image/jpeg'  # 假设图片为 JPEG，如果支持其他格式可以更改
        )

        # 调用 API
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt_text, image_part],
        )
        return response.text
    except Exception as e:
        return f"API 调用失败，错误信息：{e}"


# --- Streamlit 界面主体 ---
st.title("📄 智能发票/单据数据提取器 Demo")
st.markdown("---")

# 侧边栏：提示词配置
with st.sidebar:
    st.header("🎯 提示词（Prompt）配置")

    # 预设的结构化提取提示词
    default_prompt = """
    你是一位专业的数据分析师。请分析用户提供的图片中的发票或收据信息。
    请提取以下关键信息，并严格使用 Markdown 格式的表格输出：
    1. 购买日期 (Date)
    2. 商家名称 (Vendor Name)
    3. 总金额 (Total Amount)
    4. 具体的商品名称及数量 (Line Items)

    如果任何信息不存在，请在表格中填写 **[N/A]**。
    请在表格前面用 **## 提取结果** 作为一个二级标题。
    """

    user_prompt = st.text_area(
        "自定义提示词：",
        value=default_prompt,
        height=300,
        help="你可以修改这个提示词，让模型提取任何你想要的结构化数据。"
    )

# 主区域：文件上传
uploaded_file = st.file_uploader(
    "🖼️ 上传一张发票、收据或包含表格的图片 (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 显示上传的图片
    st.subheader("📸 已上传的图片预览")
    st.image(uploaded_file, use_column_width=True)
    st.markdown("---")

    # 运行按钮
    if st.button("启动数据提取", type="primary"):
        # 将上传的文件对象转化为 BytesIO
        image_bytes = io.BytesIO(uploaded_file.getvalue())

        # 调用核心函数
        result_text = extract_data_from_image(image_bytes, user_prompt)

        st.subheader("📝 Gemini 模型输出结果")
        # 使用 st.markdown 渲染结果，这样表格和格式会被正确显示
        st.markdown(result_text)