import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestClassifier

# --- App Configuration & Theming ---
st.set_page_config(page_title="AI Smart Predictor", page_icon="📱", layout="centered")

# --- CUSTOM CSS FOR MOBILE-FRIENDLY LOOK ---
st.markdown("""
    <style>
    /* Main Background & Padding Tweaks for Mobile */
    .stApp { background-color: #0E1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Smart Title */
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00F4B4, #0088FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #A0AEC0;
        font-size: 14px;
        margin-bottom: 20px;
    }
    
    /* Compact Cards for Mobile */
    .smart-card {
        background: #1A1C24;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        border: 1px solid #2D3748;
        margin-bottom: 15px;
    }
    
    .digit-text {
        font-size: 38px;
        font-weight: 900;
        color: white;
        margin: 5px 0;
        letter-spacing: 2px;
    }
    
    /* Progress Bar Base */
    .meter-bg {
        width: 100%;
        background-color: #2D3748;
        border-radius: 10px;
        height: 6px;
        margin-top: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📱 AI Smart Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning Auto-Predictor</div>", unsafe_allow_html=True)

DAY_MAP = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

# --- MOBILE FRIENDLY FILE UPLOAD (URL REMOVED) ---
with st.expander("📂 Data Setup (Upload File)", expanded=True):
    uploaded_file = st.file_uploader("Upload apni Excel file (.xlsx)", type=["xlsx"])
    excel_data = None
    if uploaded_file:
        excel_data = io.BytesIO(uploaded_file.getvalue())
        st.success("✅ File Loaded!")

# --- ML PROCESSING FUNCTION ---
@st.cache_resource
def process_and_train(file_bytes, sheet_name):
    try:
        df = pd.read_excel(file_bytes, sheet_name=sheet_name)
        if 'Date' not in df.columns: return None, None, None, None, None, "❌ 'Date' missing!"
            
        available_days = [col for col in df.columns if col != 'Date']
        seq_data = []
        for index, row in df.iterrows():
            for day in available_days:
                val = str(row[day]).strip()
                if val.isdigit() and len(val) == 2:
                    seq_data.append({'DayStr': day, 'DayNum': DAY_MAP.get(day, -1), 'Open': int(val[0]), 'Close': int(val[1])})
        
        data = pd.DataFrame(seq_data)
        data['Next_Open'] = data['Open'].shift(-1)
        data['Next_Close'] = data['Close'].shift(-1)
        
        last_record = data.iloc[-1].copy()
        train_data = data.dropna()
        
        X = train_data[['DayNum', 'Open', 'Close']]
        y_open = train_data['Next_Open'].astype(int)
        y_close = train_data['Next_Close'].astype(int)
        
        model_open = RandomForestClassifier(n_estimators=100, random_state=42)
        model_close = RandomForestClassifier(n_estimators=100, random_state=42)
        model_open.fit(X, y_open)
        model_close.fit(X, y_close)
        
        return data, last_record, available_days, model_open, model_close, None
    except Exception as e:
        return None, None, None, None, None, str(e)

# --- MAIN UI LOGIC ---
if excel_data is not None:
    try:
        xls = pd.ExcelFile(excel_data)
        valid_sheets = [sheet for sheet in xls.sheet_names if not sheet.lower().startswith('sheet')]
        
        market_choice = st.selectbox("🎯 Select Market:", valid_sheets if valid_sheets else xls.sheet_names)
        
        with st.spinner('🤖 AI Training in progress...'):
            data, last_record, days_order, model_open, model_close, error = process_and_train(excel_data, market_choice)
        
        if error: st.error(error)
        elif data is not None:
            last_open, last_close, last_day = int(last_record['Open']), int(last_record['Close']), last_record['DayStr']
            
            try:
                next_day_target = days_order[(days_order.index(last_day) + 1) % len(days_order)]
            except:
                next_day_target = days_order[0]
            
            # Machine Learning Prediction
            X_new = pd.DataFrame([[DAY_MAP.get(next_day_target, 0), last_open, last_close]], columns=['DayNum', 'Open', 'Close'])
            pred_o_probs, pred_c_probs = model_open.predict_proba(X_new)[0], model_close.predict_proba(X_new)[0]
            
            top_o_idx, top_c_idx = np.argsort(pred_o_probs)[::-1][:2], np.argsort(pred_c_probs)[::-1][:2]
            top_o_digits, top_c_digits = model_open.classes_[top_o_idx], model_close.classes_[top_c_idx]
            
            # Confidence Calculation
            o_conf_1 = int(pred_o_probs[top_o_idx[0]] * 100)
            c_conf_1 = int(pred_c_probs[top_c_idx[0]] * 100)
            
            # --- UI PRESENTATION FOR MOBILE ---
            st.markdown(f"<h4 style='text-align: center; color: #FFF; margin-top: 15px;'>Prediction: {next_day_target.upper()}</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            # OPEN CARD
            with col1:
                st.markdown(
                    f"<div class='smart-card' style='border-top: 3px solid #FF4B4B;'>"
                    f"<p style='color:#FF4B4B; font-size:12px; font-weight:bold; margin:0;'>🔥 OPEN</p>"
                    f"<p class='digit-text'>{top_o_digits[0]} <span style='font-size:20px; color:#4A5568;'>&</span> {top_o_digits[1]}</p>"
                    f"<div class='meter-bg'><div style='width: {o_conf_1}%; background-color: #FF4B4B; height: 100%;'></div></div>"
                    f"<p style='font-size:10px; color:gray; text-align:right; margin:2px 0 0 0;'>AI Power: {o_conf_1}%</p>"
                    f"</div>", unsafe_allow_html=True
                )
                
            # CLOSE CARD
            with col2:
                st.markdown(
                    f"<div class='smart-card' style='border-top: 3px solid #00F4B4;'>"
                    f"<p style='color:#00F4B4; font-size:12px; font-weight:bold; margin:0;'>⚡ CLOSE</p>"
                    f"<p class='digit-text'>{top_c_digits[0]} <span style='font-size:20px; color:#4A5568;'>&</span> {top_c_digits[1]}</p>"
                    f"<div class='meter-bg'><div style='width: {c_conf_1}%; background-color: #00F4B4; height: 100%;'></div></div>"
                    f"<p style='font-size:10px; color:gray; text-align:right; margin:2px 0 0 0;'>AI Power: {c_conf_1}%</p>"
                    f"</div>", unsafe_allow_html=True
                )
            
            # VIP JODI CARD
            expected_jodis = [f"{o}{c}" for o in top_o_digits for c in top_c_digits]
            st.markdown(
                f"<div class='smart-card' style='border: 1px solid #FFD700; background:rgba(255, 215, 0, 0.05);'>"
                f"<p style='color:#FFD700; font-size:12px; font-weight:bold; margin:0;'>💎 VIP JODIS</p>"
                f"<p style='color:#FFF; font-size:24px; font-weight:800; letter-spacing:5px; margin:5px 0 0 0;'>{', '.join(expected_jodis)}</p>"
                f"</div>", unsafe_allow_html=True
            )

    except Exception as general_err:
        st.error(f"❌ Error: {general_err}")
