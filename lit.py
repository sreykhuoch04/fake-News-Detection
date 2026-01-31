
import streamlit as st
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# # --------------------------------------------------
# # Page Configuration
# # --------------------------------------------------
# st.set_page_config(
#     page_title="Fake News Detection",
#     page_icon="📰",
#     layout="wide"
# )

# # --------------------------------------------------
# # Load Model (Cached)
# # --------------------------------------------------
# @st.cache_resource
# def load_model():
#     model = RobertaForSequenceClassification.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
#     tokenizer = RobertaTokenizer.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
#     model.eval()
#     return model, tokenizer

# model, tokenizer = load_model()

# # --------------------------------------------------
# # Text Cleaning
# # --------------------------------------------------
# def clean_text(text):
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"<.*?>", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# # --------------------------------------------------
# # Clear Text Callback (SAFE)
# # --------------------------------------------------
# def clear_text():
#     st.session_state.news_input = ""

# # --------------------------------------------------
# # Session State Initialization
# # --------------------------------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# if "news_input" not in st.session_state:
#     st.session_state.news_input = ""

# # --------------------------------------------------
# # Sidebar
# # --------------------------------------------------
# st.sidebar.title("ℹ️ About")
# st.sidebar.write("""
# This system uses **RoBERTa**, a transformer-based
# deep learning model fine-tuned for **Fake News Detection**.
# """)

# st.sidebar.markdown("### 🔍 Tips")
# st.sidebar.write("""
# - Paste full news articles
# - Longer text gives better accuracy
# - Avoid headlines only
# """)


# # --------------------------------------------------
# # Main UI
# # --------------------------------------------------
# st.title("📰 Fake News Detection System")
# st.markdown("Detect whether a news article is **Fake** or **Real** using AI.")
# st.divider()

# # --------------------------------------------------
# # Text Input
# # --------------------------------------------------
# text = st.text_area(
#     "📝 Enter News Article",
#     height=220,
#     key="news_input",
#     placeholder="Paste the news article here..."
# )

# # --------------------------------------------------
# # Buttons
# # --------------------------------------------------
# col1, col2 = st.columns(2)

# with col1:
#     predict_btn = st.button("🔍 Predict", use_container_width=True)

# with col2:
#     st.button(
#         "🧹 Clear Text",
#         use_container_width=True,
#         on_click=clear_text
#     )

# # --------------------------------------------------
# # Prediction Logic
# # --------------------------------------------------
# if predict_btn and text.strip() != "":
#     cleaned_text = clean_text(text)

#     inputs = tokenizer(
#         cleaned_text,
#         padding="max_length",
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)[0]

#     fake_prob = probs[0].item() * 100
#     real_prob = probs[1].item() * 100

#     label = "Real" if real_prob > fake_prob else "Fake"

#     # Save prediction history
#     st.session_state.history.append({
#         "Text": cleaned_text[:60] + "...",
#         "Fake (%)": round(fake_prob, 2),
#         "Real (%)": round(real_prob, 2),
#         "Prediction": label
#     })

#     # --------------------------------------------------
#     # Result Display
#     # --------------------------------------------------
#     st.subheader("📊 Prediction Result")

#     if label == "Real":
#         st.success(f"🟩 **Real News**")
#     else:
#         st.error(f"🟥 **Fake News**")

#     st.write(f"**Fake Probability:** {fake_prob:.2f}%")
#     st.progress(fake_prob / 100)

#     st.write(f"**Real Probability:** {real_prob:.2f}%")
#     st.progress(real_prob / 100)
#     # --------------------------------------------------
#     # Compact Probability Bar Chart
#     # --------------------------------------------------
#     labels = ["Fake", "Real"]
#     values = [fake_prob, real_prob]
#     colors = ["#E74C3C", "#2ECC71"]

#     fig, ax = plt.subplots(figsize=(5, 2.2))
#     bars = ax.barh(labels, values, color=colors)

#     ax.set_xlim(0, 100)
#     ax.set_xlabel("Probability (%)")
#     ax.set_title("Prediction Confidence", fontsize=11)

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.grid(axis="x", linestyle="--", alpha=0.4)

#     for bar in bars:
#         width = bar.get_width()
#         ax.text(
#             width + 1,
#             bar.get_y() + bar.get_height() / 2,
#             f"{width:.1f}%",
#             va="center",
#             fontsize=10
#         )

#     st.pyplot(fig)

# # ------------------
# # --------------------------------------------------
# # Prediction History (ALWAYS VISIBLE)
# # --------------------------------------------------
# if st.session_state.history:
#     st.divider()
#     st.subheader("🕒 Prediction History")

#     history_df = pd.DataFrame(st.session_state.history)
#     st.dataframe(history_df, use_container_width=True)






# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ==================================================
# Load Model (Cached)
# ==================================================
@st.cache_resource
def load_model():
    model = RobertaForSequenceClassification.from_pretrained(
         r"C:\Users\User\Downloads\bert_model\content\token_bert"
    )
    tokenizer = RobertaTokenizer.from_pretrained(
         r"C:\Users\User\Downloads\bert_model\content\token_bert"
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ==================================================
# Utilities
# ==================================================
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clear_text():
    st.session_state.news_input = ""


import streamlit as st
import json
import os

# ==================================================
# Persistent User Storage (JSON)
# ==================================================
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


# ==================================================
# Session State Initialization (SAFE & CLEAN)
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"   # better UX

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

if "users" not in st.session_state:
    st.session_state.users = load_users()  # 🔥 persistent load

if "user_history" not in st.session_state:
    st.session_state.user_history = load_history()



# ==================================================
# AUTH PAGE (SIGN UP / LOGIN)
# ==================================================
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    .auth-card {
        background: rgba(255,255,255,0.97);
        border-radius: 22px;
        padding: 34px;
        box-shadow: 0 30px 70px rgba(0,0,0,0.45);
    }
    .auth-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #020617;
    }
    .auth-subtitle {
        font-size: 14px;
        color: #475569;
        margin-bottom: 22px;
    }
    .auth-card label {
        color: #334155;
        font-weight: 600;
    }
    .auth-card input {
        color: #020617 !important;
    }
    .auth-switch {
        color: #475569;
    }
    .auth-link {
        color: #4f46e5;
        font-weight: 700;
        cursor:
        pointer;
    }
    
    * ===== HERO TEXT HEADER ===== */
    .auth-hero {
        text-align: center;
        margin: 40px auto 30px auto;
        max-width: 720px;
    }

    .auth-hero-title {
            font-family: 'Playfair Display', serif;
            font-style: italic;
            font-weight: 800;
            font-size: 46px;
            line-height: 1.2;
            background: linear-gradient(#AA336A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

    .auth-hero-subtitle {
        margin-top: 14px;
        font-size: 16px;
        color: #AA336A;
    }

    .auth-card {
        max-width: 420px;
        margin: 0 auto;
        background: rgba(255,255,255,0.96);
        border-radius: 22px;
        padding: 34px;
        box-shadow: 0 30px 70px rgba(0,0,0,0.45);
    }


    .auth-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #020617;
    }

    .auth-subtitle {
        font-size: 14px;
        color: #475569;
        margin-bottom: 22px;
    }

    .auth-card label {
        color: #334155;
        font-weight: 600;
    }

    .auth-card input {
        color: #020617 !important;
    }

    .auth-switch {
        color: #475569;
    }

    .auth-link {
        color: #4f46e5;
        font-weight: 700;
        cursor: pointer;
    }
    .auth-card:empty {
        display: none;
    }

    
    </style>
    """, unsafe_allow_html=True)
    
    # 1️⃣ HERO SECTION (TEXT ONLY – NO BOX)
    # ===== CENTER CONTAINER =====
    container = st.container()

    with container:
    # HERO
        st.markdown("""
        <div class="auth-hero">
            <div class="auth-hero-title">
                Welcome to Fake News Detection System
            </div>
            <div class="auth-hero-subtitle">
                Analyze news credibility using AI system
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AUTH CARD
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)

        # 👉 your login / signup code here

        # st.markdown("</div>", unsafe_allow_html=True)


    # ------------------------------
    # SIGN UP MODE
    # ------------------------------
    if st.session_state.auth_mode == "signup":

        st.markdown("<div class='auth-title'>Create an account</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-subtitle'>Join Fake News Detection to analyze news credibility</div>",
            unsafe_allow_html=True
        )

        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", use_container_width=True):
            if not new_username or not new_password:
                st.warning("All fields are required")
            elif new_username in st.session_state.users:
                st.error("Username already exists")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                # 🔥 FIX: SAVE USER PERMANENTLY
                st.session_state.users[new_username] = new_password
                save_users(st.session_state.users)

                st.session_state.user_history[new_username] = []
                st.success("Account created successfully")

                st.session_state.auth_mode = "login"
                st.rerun()

        st.markdown(
            "<div class='auth-switch'>Already have an account? "
            "<span class='auth-link'>Login</span></div>",
            unsafe_allow_html=True
        )

        if st.button("Go to Login"):
            st.session_state.auth_mode = "login"
            st.rerun()

    # ------------------------------
    # LOGIN MODE
    # ------------------------------
    else:

        st.markdown("<div class='auth-title'>Welcome back</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-subtitle'>Log in to continue</div>",
            unsafe_allow_html=True
        )

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if (
                username in st.session_state.users and
                st.session_state.users[username] == password
            ):
                st.session_state.logged_in = True
                st.session_state.username = username

                if username not in st.session_state.user_history:
                    st.session_state.user_history[username] = []

                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown(
            "<div class='auth-switch'>Don’t have an account? "
            "<span class='auth-link'>Sign up</span></div>",
            unsafe_allow_html=True
        )

        if st.button("Go to Sign Up"):
            st.session_state.auth_mode = "signup"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ==================================================
# SIDEBAR (LOGGED IN)
# ==================================================
st.sidebar.title("👤 User")
st.sidebar.write(f"**Logged in as:** {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()


st.sidebar.image(
    "https://edscoop.com/wp-content/uploads/sites/4/2020/07/GettyImages-1161904323.jpg?w=1013",
    use_container_width=True,
    caption="AI-powered Fake News Detection"
)

st.sidebar.divider()
st.sidebar.markdown("### ℹ️ About")
st.sidebar.write("""
This system uses **RoBERTa**, a transformer-based deep learning model fine-tuned for **Fake News Detection**, Designed for academic & real-world demo use.
""")

st.sidebar.markdown("### 🔍 Tips") 
st.sidebar.write(""" - Paste full news articles 
                     - Longer text gives better accuracy 
                     - Avoid headlines only """)
st.markdown("""
<style>
/* ===== MAIN HEADER ===== */
.title {
    font-size: 56px;        /* BIG title */
    font-weight: 900;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 8px;
    letter-spacing: 1px;
    color: #86efac;         /* light green */
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: #bbf7d0;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Main Header
# ==================================================
st.markdown("<div class='title'>📰 Fake News Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered news credibility analysis</div>", unsafe_allow_html=True)
st.divider()

# ==================================================
# Input Card
# ==================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

text = st.text_area(
    "📝 Enter News Article",
    height=220,
    key="news_input",
    placeholder="Paste full news article here..."
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

with col2:
    st.button("🧹 Clear", use_container_width=True, on_click=clear_text)

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# Prediction Logic
# ==================================================
if predict_btn and text.strip():
    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100
    label = "Real" if real_prob > fake_prob else "Fake"

    # ==============================
    # FIXED: SAVE PER-USER HISTORY
    # ==============================
    username = st.session_state.username

    if username not in st.session_state.user_history:
        st.session_state.user_history[username] = []

    st.session_state.user_history[username].append({
        "Text": cleaned_text[:60] + "...",
        "Fake (%)": round(fake_prob, 2),
        "Real (%)": round(real_prob, 2),
        "Prediction": label
    })

    save_history(st.session_state.user_history)  # 🔥 REQUIRED FOR PERSISTENCE

    # ==================================================
    # Result Card
    # ==================================================
    st.subheader("📊 Prediction Result")

    if label == "Real":
        st.success(f"🟩 **Real News**")
    else:
        st.error(f"🟥 **Fake News**")

    st.write(f"**Fake Probability:** {fake_prob:.2f}%")
    st.progress(fake_prob / 100)

    st.write(f"**Real Probability:** {real_prob:.2f}%")
    st.progress(real_prob / 100)

    # Compact Bar Chart
    labels = ["Fake", "Real"]
    values = [fake_prob, real_prob]
    colors = ["#ef4444", "#22c55e"]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Prediction Confidence", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# User History Card
# ==================================================
user_hist = st.session_state.user_history.get(st.session_state.username, [])

if user_hist:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🕒 Your Prediction History")
    st.dataframe(pd.DataFrame(user_hist), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)















































# @st.cache_resource
# def load_model():
#     model = RobertaForSequenceClassification.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
#     tokenizer = RobertaTokenizer.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
#     model.eval()
#     return model, tokenizer

# model, tokenizer = load_model()

# def clean_text(text):
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"<.*?>", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# st.title("📰 Fake News Detection (RoBERTa)")

# text = st.text_area("Enter news article")

# if st.button("Predict"):
#     text = clean_text(text)

#     inputs = tokenizer(
#         text,
#         padding="max_length",
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)[0]

#     label = torch.argmax(probs).item()
#     confidence = probs[label].item()

#     st.write("Prediction:", "🟥 Fake" if label == 0 else "🟩 Real")
#     st.write(f"Confidence: {confidence*100:.2f}%")

