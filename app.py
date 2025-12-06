import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd

# -------------------------------
# ✅ PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Underwater Trash Detection", layout="wide")

# -------------------------------
# ✅ VISITOR COUNTER
# -------------------------------
counter_file = "visitor_count.txt"

if not os.path.exists(counter_file):
    with open(counter_file, "w") as f:
        f.write("0")

with open(counter_file, "r") as f:
    count = int(f.read())

count += 1

with open(counter_file, "w") as f:
    f.write(str(count))

# -------------------------------
# ✅ USER DATABASE
# -------------------------------
user_file = "users.csv"
if not os.path.exists(user_file):
    pd.DataFrame(columns=["Name", "Country"]).to_csv(user_file, index=False)

# -------------------------------
# ✅ FEEDBACK DATABASE
# -------------------------------
feedback_file = "feedback.csv"
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=["Name", "Country", "Feedback"]).to_csv(feedback_file, index=False)

# -------------------------------
# ✅ CSS STYLING
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #1fa2ff, #12d8fa, #a6ffcb);
}
.title-box {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.visitor-box {
    background: #000;
    color: #00ffcc;
    padding: 10px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    margin-top: 10px;
}
.image-box {
    border: 5px solid #ff9800;
    padding: 10px;
    border-radius: 15px;
    background-color: white;
}
.subtitle {
    font-size: 20px;
    font-weight: bold;
    color: #0047ab;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# ✅ HEADER
# -------------------------------
st.markdown('<div class="title-box">Ashwita Ramanavelan Science Fair Project</div>', unsafe_allow_html=True)
st.markdown(f'<div class="visitor-box">👥 Total Visitors: {count}</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your details and upload an underwater image</p>', unsafe_allow_html=True)

# -------------------------------
# ✅ STORE USER STATE
# -------------------------------
if "name" not in st.session_state:
    st.session_state["name"] = ""
if "country" not in st.session_state:
    st.session_state["country"] = ""

# -------------------------------
# ✅ USER FORM
# -------------------------------
with st.form("user_form"):
    name = st.text_input("👤 Enter Your Name")
    country = st.text_input("🌎 Enter Your Country")
    submitted = st.form_submit_button("Continue")

if submitted:
    if name and country:
        st.session_state["name"] = name
        st.session_state["country"] = country

        df = pd.read_csv(user_file)
        df.loc[len(df)] = [name, country]
        df.to_csv(user_file, index=False)

        st.success(f"Welcome {name} from {country} ✅")
    else:
        st.warning("Please enter BOTH Name and Country.")

# -------------------------------
# ✅ LOAD YOLO MODEL
# -------------------------------
MODEL_PATH = .\best.pt
model = YOLO(MODEL_PATH)

# -------------------------------
# ✅ IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📸 Upload Underwater Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# ✅ DETECTION + SIDE BY SIDE DISPLAY
# -------------------------------
if uploaded_file and st.session_state["name"] and st.session_state["country"]:

    img = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        temp.write(uploaded_file.getvalue())
        temp_path = temp.name

    results = model(temp_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-box">', unsafe_allow_html=True)
        st.image(img, caption="🌊 Original Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        for r in results:
            output = r.plot()
            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.image(output, caption="🤖 AI Detection Result", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # ✅ YAY / NAY FEEDBACK
    # -------------------------------
    st.markdown("### ✅ Was this detection accurate?")

    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button("👍 Yay"):
            df_fb = pd.read_csv(feedback_file)
            df_fb.loc[len(df_fb)] = [st.session_state["name"], st.session_state["country"], "Yay"]
            df_fb.to_csv(feedback_file, index=False)
            st.success("🎉 Thank you for the positive feedback!")

    with col_no:
        if st.button("👎 Nay"):
            df_fb = pd.read_csv(feedback_file)
            df_fb.loc[len(df_fb)] = [st.session_state["name"], st.session_state["country"], "Nay"]
            df_fb.to_csv(feedback_file, index=False)
            st.warning("💡 Thanks! We’ll use this to improve.")

elif uploaded_file:
    st.warning("⚠️ Please enter your Name and Country first!")

# -------------------------------
# ✅ GLOBAL USER DISTRIBUTION
# -------------------------------
st.markdown("## 🌍 Global User Distribution")
df_users = pd.read_csv(user_file)

if not df_users.empty:
    country_counts = df_users["Country"].value_counts()
    st.bar_chart(country_counts)
else:
    st.info("No user data yet.")

# -------------------------------
# ✅ FEEDBACK SUMMARY
# -------------------------------
st.markdown("## 📊 Feedback Summary")
df_fb = pd.read_csv(feedback_file)

if not df_fb.empty:
    feedback_counts = df_fb["Feedback"].value_counts()
    st.bar_chart(feedback_counts)
else:
    st.info("No feedback yet.")
