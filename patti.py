import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestClassifier

# --- App Configuration & Theming ---
st.set_page_config(page_title="AI Pro Predictor", page_icon="💎", layout="wide")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    /* Main Background & Font Tweaks */
    .stApp { background-color: #0E1117; }
    
    /* Glowing Title */
    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #A0AEC0;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    /* Result Cards */
    .vip-card {
        background: linear-gradient(145deg, #1A1C24, #2D3748);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        border: 1px solid #4A5568;
    }
    .open-card { border-top: 5px solid #FF4B4B; }
    .close-card { border-top: 5px solid #00F4B4; }
    
    .digit-text {
        font-size: 60px;
        font-weight: 900;
        color: white;
        margin: 0;
        line-height: 1.2;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>💎 AI ML Pro Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Random Forest Prediction Engine</div>", unsafe_allow_html=True)

DAY_MAP = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

# --- CLEAN UI: FILE UPLOAD SECTION ---
# File upload ko expander mein daala hai taaki screen clear rahe
with st.expander("📂 Step 1: Upload Data (Click to Expand)", expanded=True):
    uploaded_file = st.file_uploader("Apni Excel file (.xlsx) yahan upload karein:", type=["xlsx"])
    if uploaded_file is not None:
        excel_data = io.BytesIO(uploaded_file.getvalue())
        st.success("✅ File securely loaded into memory!")
    else:
        excel_data = None

# --- ML PROCESSING FUNCTION ---
@st.cache_resource # Caching added taaki app baar-baar lag na kare
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

# --- MAIN LOGIC & UI ---
if excel_data is not None:
    try:
        xls = pd.ExcelFile(excel_data)
        valid_sheets = [sheet for sheet in xls.sheet_names if not sheet.lower().startswith('sheet')]
        
        # Market selection moved to sidebar
        st.sidebar.markdown("### ⚙️ Engine Settings")
        market_choice = st.sidebar.selectbox("🎯 Target Market:", valid_sheets if valid_sheets else xls.sheet_names)
        
        with st.spinner('⏳ AI is analyzing thousands of trees...'):
            data, last_record, days_order, model_open, model_close, error = process_and_train(excel_data, market_choice)
        
        if error: st.error(error)
        elif data is not None:
            last_open, last_close, last_day = int(last_record['Open']), int(last_record['Close']), last_record['DayStr']
            
            try:
                next_day_target = days_order[(days_order.index(last_day) + 1) % len(days_order)]
            except:
                next_day_target = days_order[0]
            
            # Prediction
            X_new = pd.DataFrame([[DAY_MAP.get(next_day_target, 0), last_open, last_close]], columns=['DayNum', 'Open', 'Close'])
            pred_o_probs, pred_c_probs = model_open.predict_proba(X_new)[0], model_close.predict_proba(X_new)[0]
            
            top_o_idx, top_c_idx = np.argsort(pred_o_probs)[::-1][:2], np.argsort(pred_c_probs)[::-1][:2]
            top_o_digits, top_c_digits = model_open.classes_[top_o_idx], model_close.classes_[top_c_idx]
            
            # UI: Header
            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center; color: #FFF;'>🎯 Target: {next_day_target.upper()} ({market_choice})</h3>", unsafe_allow_html=True)
            
            # UI: Main Prediction Cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<div class='vip-card open-card'>"
                    f"<p style='color:#FF4B4B; font-weight:bold; letter-spacing:2px;'>🔥 PREDICTED OPEN</p>"
                    f"<p class='digit-text'>{top_o_digits[0]} <span style='font-size:30px; color:gray;'>&</span> {top_o_digits[1]}</p>"
                    f"</div>", unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div class='vip-card close-card'>"
                    f"<p style='color:#00F4B4; font-weight:bold; letter-spacing:2px;'>⚡ PREDICTED CLOSE</p>"
                    f"<p class='digit-text'>{top_c_digits[0]} <span style='font-size:30px; color:gray;'>&</span> {top_c_digits[1]}</p>"
                    f"</div>", unsafe_allow_html=True
                )
            
            # UI: Beautiful Jodis Section
            expected_jodis = [f"{o}{c}" for o in top_o_digits for c in top_c_digits]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:rgba(255, 215, 0, 0.1); border: 1px solid #FFD700; padding:15px; border-radius:10px; text-align:center;'>"
                f"<h4 style='color:#FFD700; margin:0;'>💎 VIP JODIS</h4>"
                f"<h2 style='color:#FFF; letter-spacing: 10px; margin:10px 0 0 0;'>{', '.join(expected_jodis)}</h2>"
                f"</div>", unsafe_allow_html=True
            )
            
            # UI: AI Confidence Probability Charts
            st.markdown("<br><br><h4 style='color:gray;'>📊 AI Probability Analytics (Confidence Score)</h4>", unsafe_allow_html=True)
            chart_col1, chart_col2 = st.columns(2)
            
            # Prepare data for charts
            df_o_chart = pd.DataFrame({'Probability %': pred_o_probs * 100}, index=model_open.classes_).sort_values(by='Probability %', ascending=False).head(5)
            df_c_chart = pd.DataFrame({'Probability %': pred_c_probs * 100}, index=model_close.classes_).sort_values(by='Probability %', ascending=False).head(5)
            
            with chart_col1:
                st.write("📈 **Open Confidence**")
                st.bar_chart(df_o_chart, color="#FF4B4B")
            with chart_col2:
                st.write("📈 **Close Confidence**")
                st.bar_chart(df_c_chart, color="#00F4B4")

    except Exception as general_err:
        st.error(f"❌ Error: {general_err}")
else:
    st.info("👆 Kripya upar apni file upload karein taaki dashboard activate ho.")
