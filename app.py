import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
import shap
import io
import warnings
import os
warnings.filterwarnings("ignore")

try:
    import gdown
except ImportError:
    gdown = None

# ============================
# CUSTOM CSS STYLING
# ============================
def apply_custom_css():
    st.markdown("""
        <style>
        /* Main background with modern green gradient */
        .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        }
        
        /* Sidebar styling with green theme */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #134E4A 0%, #0F766E 100%);
        }
        
        [data-testid="stSidebar"] .css-1d391kg {
            color: #E0F2F1;
        }
        
        /* Main content area with clean white background */
        .main .block-container {
            background: #FFFFFF;
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        
        /* Headers styling with green gradient */
        h1 {
            color: #065F46;
            font-weight: 900;
            font-size: 3rem !important;
            margin-bottom: 1rem;
            text-align: center;
            background: linear-gradient(135deg, #059669 0%, #10B981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(5, 150, 105, 0.1);
        }
        
        h2 {
            color: #047857;
            font-weight: 700;
            margin-top: 2rem;
            padding-bottom: 0.5rem;
            border-bottom: 4px solid #10B981;
        }
        
        h3 {
            color: #065F46;
            font-weight: 600;
        }
        
        h4 {
            color: #047857;
            font-weight: 600;
            margin-top: 1.5rem;
        }
        
        /* Regular text styling for better contrast */
        p {
            color: #FFFFFF;
        }
        
        /* Metric cards with green theme */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
            color: #059669;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #374151;
            font-weight: 600;
        }
        
        /* Primary buttons with green gradient */
        .stButton>button {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 12px;
            padding: 0.875rem 2.5rem;
            font-weight: 700;
            font-size: 1.05rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #047857 0%, #059669 100%);
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
            color: #FFFFFF !important;
        }
        
        /* Info boxes with better contrast */
        .stAlert {
            border-radius: 12px;
            border-left: 6px solid #10B981;
            background-color: #F0FDF4;
            color: #065F46;
        }
        
        /* Dataframes with modern styling */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border: 1px solid #D1FAE5;
        }
        
        /* File uploader with green theme - FIXED CONTRAST */
        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
            border: 3px dashed #10B981;
            border-radius: 16px;
            padding: 2.5rem;
        }
        
        [data-testid="stFileUploader"] label {
            color: #065F46 !important;
            font-weight: 600;
        }
        
        [data-testid="stFileUploader"] section {
            color: #374151 !important;
        }
                
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
            color: #000000 !important; 
            font-weight: 600 !important;
        }
        
        [data-testid="stFileUploader"] button {
            background-color: #059669 !important;
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
        }
        
        /* Select boxes  */
        .stSelectbox > div > div {
            border-radius: 12px;
            border: 2px solid #A7F3D0;
            background-color: #FFFFFF;
        }
        
        .stSelectbox label {
            color: #065F46 !important;
            font-weight: 600;
        }
        
        .stSelectbox [data-baseweb="select"] {
            background-color: #FFFFFF;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            color: #1F2937 !important;
            background-color: #FFFFFF;
        }
        
        /* Radio buttons in sidebar - FIXED CONTRAST */
        [data-testid="stSidebar"] .stRadio > label {
            color: #E0F2F1 !important;
            font-weight: 700;
            font-size: 1.1rem;
        }
        
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            color: #E0F2F1 !important;
        }
        
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            color: #E0F2F1 !important;
        }
        
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
            color: #E0F2F1 !important;
        }
        
        /* Download button with accent color */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
            color: #FFFFFF !important;
            border-radius: 12px;
            padding: 0.875rem 2.5rem;
            font-weight: 700;
            border: none;
            box-shadow: 0 4px 20px rgba(13, 148, 136, 0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(13, 148, 136, 0.6);
            color: #FFFFFF !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #10B981 !important;
        }
        
        /* Success boxes */
        .stSuccess {
            background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
            border-left: 6px solid #059669;
            border-radius: 12px;
            color: #065F46 !important;
        }
        
        /* Warning boxes */
        .stWarning {
            background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
            border-left: 6px solid #F59E0B;
            border-radius: 12px;
            color: #92400E !important;
        }
        
        /* Error boxes */
        .stError {
            background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
            border-left: 6px solid #DC2626;
            border-radius: 12px;
            color: #991B1B !important;
        }
        
        /* Slider styling - SHAP section */
        .stSlider {
            padding-top: 0.25rem;
            padding-bottom: 0.75rem;
        }
        

        /* Bagian track yang sudah “terisi” */
        .stSlider [data-baseweb="slider"] > div > div {
            background-color: #10B981 !important;
            border-radius: 999px !important;
            height: 0.6rem !important;
        }

        /* Knob (titik bulat yang bisa digeser) */
        .stSlider div[role="slider"] {
            background-color: #047857 !important;
            border: 2px solid #ECFDF5 !important;
            width: 18px !important;
            height: 18px !important;
            box-shadow: 0 0 6px rgba(0,0,0,0.25);
        }

        /* Label slider dan angka nilainya */
        .stSlider label {
            color: #F9FAFB !important;
            font-weight: 600;
            font-size: 0.95rem;
        }

        /* Angka value slider */
        .stSlider span {
            color: #FFFFFF !important;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        /* Sidebar text and elements - FIXED CONTRAST */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #E0F2F1 !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #F0FDF4;
            border-radius: 8px;
            color: #065F46 !important;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%);
            color: white !important;
        }
        
        /* Info message styling */
        .stInfo {
            background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
            border-left: 6px solid #3B82F6;
            border-radius: 12px;
            color: #1E40AF !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #F0FDF4;
            border-radius: 8px;
            color: #065F46 !important;
            font-weight: 600;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #F9FAFB;
            border: 1px solid #D1FAE5;
            border-radius: 8px;
        }
        
        /* Markdown text in main area */
        .main .block-container p,
        .main .block-container li,
        .main .block-container span {
            color: #374151;
        }
        
        /* Spinner text */
        .stSpinner > div > div {
            color: #059669 !important;
        }
        
        /* Table header and cells - better contrast */
        thead tr th {
            background-color: #F0FDF4 !important;
            color: #065F46 !important;
            font-weight: 700;
        }
        
        tbody tr td {
            color: #1F2937 !important;
        }
        
        /* Input fields */
        input, textarea, select {
            color: #1F2937 !important;
            background-color: #FFFFFF !important;
        }
        
        /* Number input */
        .stNumberInput input {
            color: #1F2937 !important;
        }
        
        /* Text input */
        .stTextInput input {
            color: #1F2937 !important;
        }
        
        /* Text area */
        .stTextArea textarea {
            color: #1F2937 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================
# MODEL CONFIG
# ============================
MODEL_DIR = "models"
MODEL_FILENAME = "bert_bigru_model.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

DRIVE_URL = "https://drive.google.com/uc?id=1zD65iGiOA0aHqCQe0sYbAv9L0S3bbLqJ"

# ============================
# MODEL DEFINITION
# ============================
class BertBiGRUClassifier(nn.Module):
    def __init__(self, bert_model, gru_hidden, num_classes=3, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size
        self.bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(gru_hidden * 2)
        self.fc = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        _, hidden = self.bigru(last_hidden_state)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.layer_norm(hidden_cat)
        out = self.dropout(out)
        return self.fc(out)

# ============================
# EMOJI MAPPING
# ============================
@st.cache_data
def load_emoji_map():
    try:
        emoji_df = pd.read_csv("emoji_to_text.csv", encoding="utf-8")
        return dict(zip(emoji_df.iloc[:, 0], emoji_df.iloc[:, 1]))
    except:
        return {}

emoji_map = load_emoji_map()

def replace_emojis(text):
    text = str(text)
    for emo, meaning in emoji_map.items():
        if emo in text:
            text = text.replace(emo, f'[{meaning}]')
    return text

# ============================
# PREPROCESSING FUNCTION
# ============================
def preprocessing(text):
    """Preprocessing teks sesuai dengan training"""
    text = replace_emojis(text)
    text = re.sub(r'@', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r"http\S+", '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================
# LOAD MODEL & TOKENIZER
# ============================
@st.cache_resource
def load_model_and_tokenizer():
    """Load model dan tokenizer (cached)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

    # Pastikan file model ada di folder lokal (models/bert_bigru_model.pt)
    if not os.path.exists(MODEL_PATH):
        # Kalau belum ada, coba auto-download dari Google Drive
        if gdown is None:
            # Kalau gdown belum terinstall / tidak tersedia, berikan pesan jelas
            raise FileNotFoundError(
                f"Model tidak ditemukan di {MODEL_PATH}. "
                "Silakan download file model dari Google Drive dan simpan di folder 'models/'."
            )
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

    # Load backbone IndoBERTweet
    bert_model = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")

    # Inisialisasi arsitektur IndoBERTweet-BiGRU
    model = BertBiGRUClassifier(
        bert_model,
        gru_hidden=128,
        dropout=0.20301892772651725,
        num_classes=3
    )

    # Load bobot terlatih dari file .pt
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, tokenizer, device

# ============================
# PREDICTION FUNCTION
# ============================
def predict_sentiment(texts, model, tokenizer, device, max_length=128):
    """Prediksi sentimen untuk list teks"""
    class_names = ["Positif", "Netral", "Negatif"]
    cleaned_texts = [preprocessing(text) for text in texts]
    encodings = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(encodings["input_ids"], encodings["attention_mask"])
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    
    predictions = np.argmax(probs, axis=1)
    pred_labels = [class_names[p] for p in predictions]
    max_probs = np.max(probs, axis=1)
    
    return pred_labels, max_probs, probs, cleaned_texts

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(
    page_title="Explainable AI - Analisis Sentimen Mangrove",
    page_icon="🌿",
    layout="wide"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False
if 'lime_results' not in st.session_state:
    st.session_state['lime_results'] = {}
if 'shap_computed' not in st.session_state:
    st.session_state['shap_computed'] = False

# ============================
# SIDEBAR
# ============================
st.sidebar.markdown("<h1 style='color: #E0F2F1; text-align: center; font-size: 2rem; margin-bottom: 0;'>🌿 Mangrove XAI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #99F6E4; text-align: center; margin-bottom: 2rem; font-size: 0.95rem;'>Sistem Explainable AI untuk Memahami Sentimen Publik Terkait Mangrove</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🗂️ Navigasi",
    ["Prediksi Teks", "Penjelasan Lokal dengan LIME", "Penjelasan Global dengan SHAP"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Load model
try:
    model, tokenizer, device = load_model_and_tokenizer()
    st.sidebar.markdown("""
        <div style='background: rgba(16, 185, 129, 0.2); padding: 1.2rem; border-radius: 12px; border-left: 5px solid #10B981;'>
            <p style='color: #D1FAE5; margin: 0; font-weight: 700; font-size: 0.95rem;'>✅ Model Loaded Successfully</p>
        </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ============================
# PAGE 1: PREDIKSI
# ============================
if page == "Prediksi Teks":
    st.markdown("<h1>🌿 Mangrove Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.15rem; margin-bottom: 2.5rem; font-weight: 500;'>Sistem Explainable AI untuk Memahami Sentimen Publik Terkait Mangrove</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔮 Prediksi Sentimen Otomatis")
    st.markdown("<p style='color: #E5E7EB; font-size: 1rem;'>Upload file Excel Anda untuk mendapatkan analisis sentimen secara otomatis dengan AI</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        " ",
        type=["xlsx", "xls"],
        help="Upload file Excel yang berisi kolom teks untuk dianalisis"
    )
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); 
                        padding: 1.2rem; border-radius: 12px; margin: 1.5rem 0;
                        border-left: 6px solid #059669; box-shadow: 0 2px 8px rgba(5, 150, 105, 0.15);'>
                <p style='margin: 0; font-weight: 700; color: #065F46; font-size: 1rem;'>
                    ✅ File berhasil diupload! Total baris: <strong>{}</strong></p>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
        
        st.markdown("### 📋 Preview Data")
        st.dataframe(df.head(), use_container_width=True)
        
        text_column = st.selectbox(
            "📝 Pilih kolom yang berisi teks untuk analisis:",
            options=df.columns.tolist(),
            help="Pilih kolom yang berisi teks yang ingin Anda analisis"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Mulai Prediksi", type="primary", use_container_width=True):
                with st.spinner("🔄 Memproses data dan menganalisis sentimen..."):
                    texts = df[text_column].astype(str).tolist()
                    pred_labels, max_probs, all_probs, cleaned_texts = predict_sentiment(
                        texts, model, tokenizer, device
                    )
                    
                    result_df = pd.DataFrame({
                        'Teks Asli': texts,
                        'Teks Bersih': cleaned_texts,
                        'Prediksi': pred_labels,
                        'Probabilitas': max_probs.round(4),
                        'Prob_Positif': all_probs[:, 0].round(4),
                        'Prob_Netral': all_probs[:, 1].round(4),
                        'Prob_Negatif': all_probs[:, 2].round(4)
                    })
                    
                    st.session_state['result_df'] = result_df
                    st.session_state['texts'] = texts
                    st.session_state['cleaned_texts'] = cleaned_texts
                    st.session_state['all_probs'] = all_probs
                    st.session_state['prediction_done'] = True
                    st.session_state['lime_results'] = {}
                    st.session_state['shap_computed'] = False
                
                st.success("✨ Prediksi selesai dengan sempurna!")
                st.rerun()
    
    # Display results
    if st.session_state['prediction_done'] and 'result_df' in st.session_state:
        result_df = st.session_state['result_df']
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis Sentimen")
        
        # Metrics in colored cards
        col1, col2, col3, col4 = st.columns(4)
        
        positif = (result_df['Prediksi'] == 'Positif').sum()
        netral = (result_df['Prediksi'] == 'Netral').sum()
        negatif = (result_df['Prediksi'] == 'Negatif').sum()
        
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #059669 0%, #10B981 100%); 
                            padding: 2rem 1.5rem; border-radius: 16px; text-align: center;
                            box-shadow: 0 8px 24px rgba(5, 150, 105, 0.25);'>
                    <p style='color: #D1FAE5; font-size: 0.9rem; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Total Data</p>
                    <p style='color: #FFFFFF; font-size: 3rem; font-weight: 900; margin: 0.5rem 0;'>{len(result_df)}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%); 
                            padding: 2rem 1.5rem; border-radius: 16px; text-align: center;
                            box-shadow: 0 8px 24px rgba(13, 148, 136, 0.25);'>
                    <p style='color: #CCFBF1; font-size: 0.9rem; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>😊 Positif</p>
                    <p style='color: #FFFFFF; font-size: 3rem; font-weight: 900; margin: 0.5rem 0;'>{positif}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #0891B2 0%, #06B6D4 100%); 
                            padding: 2rem 1.5rem; border-radius: 16px; text-align: center;
                            box-shadow: 0 8px 24px rgba(8, 145, 178, 0.25);'>
                    <p style='color: #CFFAFE; font-size: 0.9rem; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>😐 Netral</p>
                    <p style='color: #FFFFFF; font-size: 3rem; font-weight: 900; margin: 0.5rem 0;'>{netral}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%); 
                            padding: 2rem 1.5rem; border-radius: 16px; text-align: center;
                            box-shadow: 0 8px 24px rgba(220, 38, 38, 0.25);'>
                    <p style='color: #FEE2E2; font-size: 0.9rem; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>😞 Negatif</p>
                    <p style='color: #FFFFFF; font-size: 3rem; font-weight: 900; margin: 0.5rem 0;'>{negatif}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results table
        st.markdown("#### 📝 Detail Hasil Prediksi")
        st.dataframe(
            result_df[['Teks Asli', 'Prediksi', 'Probabilitas']],
            use_container_width=True,
            height=400
        )
        
        # Download button
        st.markdown("---")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Hasil Prediksi (Excel)",
                data=output.getvalue(),
                file_name="hasil_prediksi_sentimen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# ============================
# PAGE 2: LIME
# ============================
elif page == "Penjelasan Lokal dengan LIME":
    st.markdown("<h1>🌿 Mangrove Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.15rem; margin-bottom: 2.5rem; font-weight: 500;'>Sistem Explainable AI untuk Memahami Sentimen Publik Terkait Mangrove</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔍 LIME - Local Interpretable Model-agnostic Explanations")
    st.markdown("<p style='color: #E5E7EB; font-size: 1rem;'>Analisis kata-kata spesifik yang mempengaruhi prediksi model untuk setiap teks</p>", unsafe_allow_html=True)
    
    if not st.session_state['prediction_done'] or 'result_df' not in st.session_state:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
                        padding: 2rem; border-radius: 12px; margin: 2rem 0;
                        border-left: 6px solid #F59E0B; text-align: center;
                        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);'>
                <p style='margin: 0; font-weight: 700; color: #92400E; font-size: 1.15rem;'>
                    ⚠️ Silakan upload dan prediksi data terlebih dahulu di halaman <strong>Prediksi</strong></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        result_df = st.session_state['result_df']
        texts = st.session_state['texts']
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); 
                        padding: 1.2rem; border-radius: 12px; margin: 1.5rem 0;
                        border-left: 6px solid #0891B2; box-shadow: 0 2px 8px rgba(8, 145, 178, 0.15);'>
                <p style='margin: 0; font-weight: 700; color: #0C4A6E;'>
                    📊 Total data tersedia untuk analisis: <strong>{len(result_df)}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Select sample
        st.markdown("#### 🎯 Pilih Teks untuk Dianalisis")
        selected_idx = st.selectbox(
            "Pilih index teks:",
            options=range(len(result_df)),
            format_func=lambda x: f"Index {x}: {result_df['Teks Asli'].iloc[x][:80]}...",
            label_visibility="collapsed"
        )
        
        # Display sample info
        sample_text = texts[selected_idx]
        sample_pred = result_df['Prediksi'].iloc[selected_idx]
        sample_prob = result_df['Probabilitas'].iloc[selected_idx]
        
        st.markdown("#### 📝 Informasi Teks Terpilih")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); 
                            padding: 2rem; border-radius: 12px; 
                            border-left: 6px solid #059669; height: 100%;
                            box-shadow: 0 2px 8px rgba(5, 150, 105, 0.1);'>
                    <p style='font-weight: 700; color: #065F46; margin-bottom: 1rem; font-size: 1.05rem;'>📄 Teks Asli:</p>
                    <p style='color: #374151; line-height: 1.7; font-size: 0.95rem;'>{}</p>
                </div>
            """.format(sample_text), unsafe_allow_html=True)
        
        with col2:
            pred_color = {'Positif': '#0D9488', 'Netral': '#0891B2', 'Negatif': '#DC2626'}
            st.markdown("""
                <div style='background: linear-gradient(135deg, {} 0%, {}dd 100%); 
                            padding: 2rem; border-radius: 12px; 
                            height: 100%; display: flex; flex-direction: column; 
                            justify-content: center; text-align: center;
                            box-shadow: 0 4px 16px rgba(0,0,0,0.15);'>
                    <p style='font-weight: 700; color: white; margin-bottom: 0.5rem; 
                              font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;'>🎯 Prediksi Model</p>
                    <p style='color: white; font-size: 2.5rem; font-weight: 900; margin: 0.5rem 0;'>{}</p>
                    <p style='color: rgba(255,255,255,0.95); font-size: 1.1rem; margin-top: 0.5rem; font-weight: 600;'>
                        Probabilitas: {:.2%}</p>
                </div>
            """.format(pred_color[sample_pred], pred_color[sample_pred], sample_pred, sample_prob), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analisis dengan LIME", type="primary", use_container_width=True):
                with st.spinner("🔄 Menghitung penjelasan LIME..."):
                    def predict_proba_lime(texts):
                        model.eval()
                        cleaned = [preprocessing(t) for t in texts]
                        enc = tokenizer(
                            cleaned,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        ).to(device)
                        with torch.no_grad():
                            outputs = model(enc["input_ids"], enc["attention_mask"])
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        return probs
                    
                    class_names = ["Positif", "Netral", "Negatif"]
                    explainer = LimeTextExplainer(class_names=class_names, random_state=42)
                    pred_class = class_names.index(sample_pred)
                    exp = explainer.explain_instance(
                        sample_text,
                        predict_proba_lime,
                        num_features=10,
                        labels=[pred_class],
                        num_samples=200
                    )
                    
                    st.session_state['lime_results'][selected_idx] = {
                        'exp': exp,
                        'html': exp.as_html(),
                        'pred_class': pred_class,
                        'class_names': class_names
                    }
                
                st.success("✨ Analisis LIME selesai!")
                st.rerun()
        
        # Display LIME results
        if selected_idx in st.session_state['lime_results']:
            lime_data = st.session_state['lime_results'][selected_idx]
            
            st.markdown("---")
            st.markdown("#### 🎨 Visualisasi LIME")
            st.markdown("""
                <div style='background: linear-gradient(135deg, #F0FDF4 0%, #D1FAE5 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                            border-left: 6px solid #10B981; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);'>
                    <p style='margin: 0; color: #065F46; text-align: center; font-weight: 600; font-size: 1rem;'>
                        💡 <strong>Interpretasi:</strong> Kata disebelah kanan mendukung prediksi, 
                        kata disebelah kiri menentang prediksi. Semakin pekat warna highlight, semakin besar pengaruhnya.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            lime_html = lime_data['html']

            custom_css = """
            <style>
            /* Semua teks HTML jadi putih */
            * {
                color: #FFFFFF !important;
            }
            body {
                background-color: transparent;
            }
            /* Teks di dalam grafik LIME (SVG) jadi putih juga */
            svg text {
                fill: #FFFFFF !important;
            }
            </style>
            """

            # sisipkan CSS sebelum </head>
            lime_html = lime_html.replace("</head>", custom_css + "</head>")

            st.components.v1.html(lime_html, height=400, scrolling=True)
            
            # st.components.v1.html(lime_data['html'], height=400, scrolling=True)

# ============================
# PAGE 3: SHAP
# ============================
elif page == "Penjelasan Global dengan SHAP":
    st.markdown("<h1>🌿 Mangrove Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.15rem; margin-bottom: 2.5rem; font-weight: 500;'>Sistem Explainable AI untuk Memahami Sentimen Publik Terkait Mangrove</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📈 SHAP - SHapley Additive exPlanations")
    st.markdown("<p style='color: #E5E7EB; font-size: 1rem;'>Analisis fitur-fitur yang paling berpengaruh secara keseluruhan terhadap model</p>", unsafe_allow_html=True)
    
    if not st.session_state['prediction_done'] or 'result_df' not in st.session_state:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
                        padding: 2rem; border-radius: 12px; margin: 2rem 0;
                        border-left: 6px solid #F59E0B; text-align: center;
                        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);'>
                <p style='margin: 0; font-weight: 700; color: #92400E; font-size: 1.15rem;'>
                    ⚠️ Silakan upload dan prediksi data terlebih dahulu di halaman <strong>Prediksi</strong></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        cleaned_texts = st.session_state['cleaned_texts']
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); 
                        padding: 1.2rem; border-radius: 12px; margin: 1.5rem 0;
                        border-left: 6px solid #0891B2; box-shadow: 0 2px 8px rgba(8, 145, 178, 0.15);'>
                <p style='margin: 0; font-weight: 700; color: #0C4A6E;'>
                    📊 Total data tersedia untuk analisis SHAP: <strong>{len(cleaned_texts)}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### ⚙️ Konfigurasi Analisis")
        st.markdown("<p style='color: #FFFFFF; font-size: 0.95rem;'>Pilih jumlah sampel yang akan dianalisis</p>", unsafe_allow_html=True)
        
        max_samples = st.slider(
            "Jumlah sampel untuk analisis SHAP:",
            min_value=10,
            max_value=min(500, len(cleaned_texts)),
            value=min(100, len(cleaned_texts)),
            step=10,
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%); 
                        padding: 1.2rem; border-radius: 10px; margin: 1.5rem 0;
                        border-left: 4px solid #059669;'>
                <p style='margin: 0; color: #374151; font-weight: 600;'>
                    📌 <strong style='color: #059669;'>{max_samples}</strong> sampel akan dianalisis
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Hitung SHAP Values", type="primary", use_container_width=True):
                with st.spinner(f"🔄 Menghitung SHAP values untuk {max_samples} sampel... Mohon tunggu."):
                    sample_texts = cleaned_texts[:max_samples]
                    
                    def shap_predict(texts):
                        if isinstance(texts, np.ndarray):
                            texts = texts.tolist()
                        if isinstance(texts, str):
                            texts = [texts]
                        elif isinstance(texts, (list, tuple)) and len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
                            texts = [" ".join(map(str, t)) for t in texts]
                        
                        enc = tokenizer(
                            texts,
                            padding=True,
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        ).to(device)
                        
                        with torch.no_grad():
                            outputs = model(enc["input_ids"], enc["attention_mask"])
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        return probs
                    
                    masker = shap.maskers.Text(r"\W")
                    explainer = shap.Explainer(shap_predict, masker, seed=42)
                    shap_values = explainer(sample_texts)
                    
                    st.session_state['shap_values'] = shap_values
                    st.session_state['shap_texts'] = sample_texts
                    st.session_state['shap_computed'] = True
                
                st.success("✨ Perhitungan SHAP selesai dengan sempurna!")
                st.rerun()
        
        # Display SHAP results
        if st.session_state['shap_computed'] and 'shap_values' in st.session_state:
            shap_values = st.session_state['shap_values']
            sample_texts = st.session_state['shap_texts']
            
            st.markdown("---")
            st.markdown("### 📊 Global Feature Importance Analysis")
            st.markdown("""
                <div style='background: linear-gradient(135deg, #F0FDF4 0%, #D1FAE5 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
                            border-left: 6px solid #10B981; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);'>
                    <p style='margin: 0; color: #065F46; text-align: center; font-weight: 600; font-size: 1rem;'>
                        💡 <strong>Interpretasi:</strong> Token dengan nilai SHAP lebih tinggi memiliki 
                        pengaruh lebih besar terhadap prediksi kelas tersebut
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            class_names = ["Positif", "Netral", "Negatif"]
            class_colors = ['#0D9488', '#0891B2', '#DC2626']

            # Tampilkan tiga visualisasi dalam satu baris
            cols = st.columns(3)

            for class_idx, (class_name, class_color, col) in enumerate(zip(class_names, class_colors, cols)):
                with col:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {class_color} 0%, {class_color}dd 100%); 
                                    padding: 1.2rem; border-radius: 12px; margin: 1rem 0;
                                    box-shadow: 0 4px 16px rgba(0,0,0,0.15);'>
                            <h3 style='color: white; margin: 0; text-align: center; font-size: 1.2rem; font-weight: 800;'>
                                🎯 {class_name}
                            </h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Aggregate token importance
                    token_scores = {}
                    for i in range(len(sample_texts)):
                        tokens = shap_values.data[i]
                        vals = shap_values.values[i]

                        # Jika shape (token, num_class), ambil kelas tertentu
                        if vals.ndim == 2:
                            vals = np.abs(vals[:, class_idx])
                        else:
                            vals = np.abs(vals)

                        for t, s in zip(tokens, vals):
                            if t not in token_scores:
                                token_scores[t] = []
                            token_scores[t].append(float(s))

                    # Rata-rata importance tiap token
                    avg_importance = {t: np.mean(vs) for t, vs in token_scores.items()}
                    top_items = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]

                    if not top_items:
                        st.info("Belum ada token penting yang terdeteksi untuk kelas ini.")
                        continue

                    words, scores = zip(*top_items)

                    # Figur diperkecil & kompak
                    fig, ax = plt.subplots(figsize=(5, 4))  # lebih kecil, satu baris enak
                    y_pos = np.arange(len(words))

                    bars = ax.barh(
                        y_pos,
                        scores,
                        edgecolor='white',
                        linewidth=1.5
                    )

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words, fontsize=9, fontweight='600', color='#1F2937')

                    ax.set_xlabel('|SHAP Value| (rata-rata)', fontsize=9, fontweight='bold', color='#065F46')
                    ax.set_title(
                        f'Top Token\n{class_name}',
                        fontsize=10,
                        fontweight='bold',
                        color='#065F46',
                        pad=10
                    )

                    # Supaya token paling berpengaruh di paling atas
                    ax.invert_yaxis()

                    # Styling
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_color('#D1D5DB')
                    ax.tick_params(left=False, colors='#374151')
                    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=1)
                    ax.set_axisbelow(True)

                    # Label nilai di ujung bar
                    for bar, score in zip(bars, scores):
                        width = bar.get_width()
                        ax.text(
                            width,
                            bar.get_y() + bar.get_height() / 2,
                            f' {score:.4f}',
                            ha='left',
                            va='center',
                            fontsize=8,
                            fontweight='600',
                            color='#111827'
                        )

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                if class_idx < len(class_names) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)