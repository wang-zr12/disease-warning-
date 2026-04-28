import streamlit as st
import requests
import json

st.title("NHANES Prediction Frontend")

st.write("输入 JSON 格式的特征数据，点击提交以测试后端。")

default_input = {
    "age": 60,
    "blood_pressure": 120,
    "creatinine": 1.1
}

input_text = st.text_area("输入 JSON：", json.dumps(default_input, indent=2), height=200)

if st.button("提交预测请求"):
    try:
        data = json.loads(input_text)
        with st.spinner("正在发送到 Gateway..."):
            resp = requests.post("http://10.0.0.35:8080/predict_ckd", json=data)
        st.success("✅ 返回结果：lala")
        st.json(resp.json())
    except Exception as e:
        st.error(f"请求失败: {e}")
