import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import plotly.express as px
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------------------
# Page Configuration & Styling
# ------------------------------
st.set_page_config(page_title="AI Resume Intelligence", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .status-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Load Model (with Spinner)
# ------------------------------
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# ------------------------------
# Modern Job Role Definitions
# ------------------------------
job_roles = {
    "Data Scientist": "python machine learning sql statistics pandas scikit-learn visualization deep learning",
    "Web Developer": "html css javascript react nodejs typescript tailwind backend frontend api",
    "Software Engineer": "java python c++ system design distributed systems docker kubernetes algorithms",
    "ML Engineer": "pytorch tensorflow keras mlops model deployment scikit-learn deep learning transformers",
    "Business Analyst": "excel tableau powerbi sql requirements gathering agile stakeholder management reporting",
    "Cybersecurity": "siem firewalls penetration testing encryption linux network security incident response"
}

# ------------------------------
# Logic Functions
# ------------------------------
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def get_analysis(text):
    # NLP Classification
    labels = list(job_roles.keys())
    nlp_result = classifier(text[:1000], labels)
    
    # TF-IDF Similarity
    corpus = [text] + list(job_roles.values())
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(corpus)
    sim_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    return nlp_result, sim_scores

# ------------------------------
# UI Layout
# ------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942799.png", width=100)
    st.title("Settings")
    st.info("This AI analyzes your resume against industry benchmarks using Zero-Shot Classification and Cosine Similarity.")
    threshold = st.slider("Match Sensitivity", 0.0, 1.0, 0.4)

st.title("💼 Smart Resume Intelligence")
st.write("Upload your PDF resume to see how you rank against top industry roles.")

uploaded_file = st.file_uploader("Drop your resume here", type=["pdf"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with st.spinner("🧠 AI is reading your resume..."):
        resume_text = extract_text(uploaded_file)
        nlp_res, sim_res = get_analysis(resume_text)
    
    with col1:
        st.subheader("🎯 Role Match Probability")
        # Create a DataFrame for the chart
        df = pd.DataFrame({
            "Role": nlp_res['labels'],
            "Confidence": nlp_res['scores']
        }).sort_values("Confidence", ascending=True)
        
        fig = px.bar(df, x="Confidence", y="Role", orientation='h', 
                     color="Confidence", color_continuous_scale="RdYlGn")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 Recommendation")
        best_role = nlp_res['labels'][0]
        score = nlp_res['scores'][0]
        
        if score > threshold:
            st.success(f"### High Match: {best_role}")
            st.metric("AI Confidence Score", f"{score*100:.1f}%")
            st.progress(score)
            st.write("✨ Your profile shows strong alignment with this role's core requirements.")
        else:
            st.warning("No strong match found. Consider tailoring your resume with more specific keywords.")

    st.divider()
    
    # Keyword Comparison Section
    st.subheader("🔍 Skillset Deep Dive")
    tabs = st.tabs(list(job_roles.keys()))
    
    for i, role in enumerate(job_roles.keys()):
        with tabs[i]:
            sim_val = sim_res[i]
            st.write(f"**Similarity with {role} Keywords:** {sim_val:.2f}")
            st.progress(float(sim_val))
            st.caption(f"Required Keywords: {job_roles[role]}")

else:
    st.image("https://illustrations.popsy.co/gray/hiring.svg", width=400)