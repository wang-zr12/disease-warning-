import streamlit as st
import requests

# 1. Render 后端 API 的公网地址
API_BASE_URL = "https://disease-warning.onrender.com"

# --- Streamlit 页面布局 ---
st.title("后端 API 完整测试页面")
st.write(f"正在测试的后端: `{API_BASE_URL}`")

# --- 共享的输入控件 ---
st.header("共享输入数据")
st.write("下面的控件将用于所有的 API 测试。")

# 用于 /metrics/{disease} 和两个 POST 请求
disease_to_predict = st.selectbox(
    "选择疾病 (用于所有请求)",
    ["Heart Disease", "Diabetes", "Chronic Kidney Disease"]
)

# 用于两个 POST 请求
example_input_data = {
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 250,
}
st.subheader("将用于 POST 请求的 'input_data':")
st.json(example_input_data)

# 准备所有 POST 请求都需要的主体 (Body)
data_to_send = {
    "disease": disease_to_predict,
    "input_data": example_input_data
}

st.divider()

# --- 测试 1: Prediction API ---
st.header("1. 测试: `POST /prediction/`")

if st.button("开始预测"):
    predict_endpoint = f"{API_BASE_URL}/prediction/"

    try:
        st.write(f"正在向 `{predict_endpoint}` 发送 POST 请求...")
        st.write("发送的 JSON Body:")
        st.json(data_to_send)

        response = requests.post(predict_endpoint, json=data_to_send)

        st.write(f"**收到的状态码: {response.status_code}**")
        if response.status_code == 200:
            st.success("预测成功！🎉")
            st.subheader("收到的结果:")
            st.json(response.json())
        else:
            st.error("API 请求失败")
            st.subheader("收到的错误详情:")
            try:
                st.json(response.json())
            except requests.exceptions.JSONDecodeError:
                st.text(response.text)

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接失败: {e}")
        st.write("请检查后端服务是否正在运行。")

st.divider()

# --- 测试 2: Visualization API ---
st.header("2. 测试: `POST /visualization/`")

if st.button("获取可视化数据"):
    viz_endpoint = f"{API_BASE_URL}/visualization/"

    try:
        st.write(f"正在向 `{viz_endpoint}` 发送 POST 请求...")
        st.write("发送的 JSON Body:")
        st.json(data_to_send)

        response = requests.post(viz_endpoint, json=data_to_send)

        st.write(f"**收到的状态码: {response.status_code}**")
        if response.status_code == 200:
            st.success("获取数据成功！🎉")
            st.subheader("收到的结果 (图表等):")
            st.json(response.json())
        else:
            st.error("API 请求失败")
            st.subheader("收到的错误详情:")
            try:
                st.json(response.json())
            except requests.exceptions.JSONDecodeError:
                st.text(response.text)

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接失败: {e}")

st.divider()

# --- 测试 3: Metrics API ---
st.header("3. 测试: `GET /metrics/{disease}`")

if st.button("获取 Metrics"):
    # 这个 API 使用了路径参数 (Path Parameter)
    metrics_endpoint = f"{API_BASE_URL}/metrics/{disease_to_predict}"

    try:
        st.write(f"正在向 `{metrics_endpoint}` 发送 GET 请求...")

        response = requests.get(metrics_endpoint)

        st.write(f"**收到的状态码: {response.status_code}**")
        if response.status_code == 200:
            st.success("获取 Metrics 成功！🎉")
            st.subheader("收到的结果:")
            st.json(response.json())
        else:
            st.error("API 请求失败")
            st.subheader("收到的错误详情:")
            try:
                st.json(response.json())
            except requests.exceptions.JSONDecodeError:
                st.text(response.text)

    except requests.exceptions.ConnectionError as e:
        st.error(f"连接失败: {e}")