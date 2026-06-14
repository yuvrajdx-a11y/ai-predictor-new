import streamlit as st
import pandas as pd
import numpy as np
import io # Nayi library add ki hai error fix karne ke liye
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI ML Training Engine", page_icon="🧠", layout="wide")

st.title("🧠 AI Machine Learning Training Engine")
st.markdown("Yeh script basic math par nahi, balki **Random Forest Machine Learning Algorithm** par aadharit hai. Yeh aapke pichle data par khud ko 'Train' karti hai aur agle draw ka prediction karti hai.")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Kripya apni Excel file (.xlsx) yahan upload karein:", type=["xlsx"])

# Day ko number mein badalne ke liye dictionary
DAY_MAP = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

def process_and_train(file_bytes, sheet_name):
    try:
        df = pd.read_excel(file_bytes, sheet_name=sheet_name)
        if 'Date' not in df.columns: return None, None, None, None, None, "❌ 'Date' column missing!"
            
        available_days = [col for col in df.columns if col != 'Date']
        
        # Data ko line se lagana
        seq_data = []
        for index, row in df.iterrows():
            for day in available_days:
                val = str(row[day]).strip()
                if val.isdigit() and len(val) == 2:
                    seq_data.append({
                        'DayStr': day,
                        'DayNum': DAY_MAP.get(day, -1),
                        'Open': int(val[0]), 
                        'Close': int(val[1])
                    })
        
        data = pd.DataFrame(seq_data)
        
        # Target variables banana (Agle din kya aaya)
        data['Next_Open'] = data['Open'].shift(-1)
        data['Next_Close'] = data['Close'].shift(-1)
        
        # Aakhiri row ko prediction ke liye alag nikalna, aur baaki par train karna
        last_record = data.iloc[-1].copy()
        train_data = data.dropna() # Nan values hata do
        
        # --- MACHINE LEARNING TRAINING PHASE ---
        # Features (X): Aaj ka din, Aaj ka Open, Aaj ka Close
        X = train_data[['DayNum', 'Open', 'Close']]
        
        # Targets (y): Agla Open, Agla Close
        y_open = train_data['Next_Open'].astype(int)
        y_close = train_data['Next_Close'].astype(int)
        
        # Models Initialize karna
        model_open = RandomForestClassifier(n_estimators=100, random_state=42)
        model_close = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Models ko data par Train karna
        model_open.fit(X, y_open)
        model_close.fit(X, y_close)
        
        return data, last_record, available_days, model_open, model_close, None
    except Exception as e:
        return None, None, None, None, None, str(e)

if uploaded_file is not None:
    try:
        # BUG FIX: Mobile se bytes upload hone wali problem theek ki gayi hai
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        xls = pd.ExcelFile(file_bytes)
        valid_sheets = [sheet for sheet in xls.sheet_names if not sheet.lower().startswith('sheet')]
        if not valid_sheets: valid_sheets = xls.sheet_names
            
        st.sidebar.header("⚙️ Settings")
        market_choice = st.sidebar.selectbox("Market Chunein:", valid_sheets)
        
        with st.spinner('🤖 AI apne aap ko data par train kar raha hai... Kripya pratiksha karein...'):
            data, last_record, days_order, model_open, model_close, error = process_and_train(file_bytes, market_choice)
        
        if error: st.error(error)
        elif data is None or data.empty: st.warning("⚠️ Valid data nahi mila.")
        else:
            last_open = int(last_record['Open'])
            last_close = int(last_record['Close'])
            last_day = last_record['DayStr']
            
            try:
                curr_idx = days_order.index(last_day)
                next_day_target = days_order[(curr_idx + 1) % len(days_order)]
            except:
                next_day_target = days_order[0]
                
            next_day_num = DAY_MAP.get(next_day_target, 0)
            
            st.success(f"✅ **Training Complete!** AI ne lagbhag {len(data)} records par pattern seekh liya hai.")
            
            # --- AI PREDICTION PHASE ---
            # Model ko naya data dena (Aakhiri record) taaki wo next day predict kare
            X_new = pd.DataFrame([[next_day_num, last_open, last_close]], columns=['DayNum', 'Open', 'Close'])
            
            # Prediction aur unki Probabilities (Confidence) nikalna
            pred_open_probs = model_open.predict_proba(X_new)[0]
            pred_close_probs = model_close.predict_proba(X_new)[0]
            
            # Sabse zyada probability wale top 2 numbers nikalna
            top_2_open_idx = np.argsort(pred_open_probs)[::-1][:2]
            top_2_close_idx = np.argsort(pred_close_probs)[::-1][:2]
            
            top_open_digits = model_open.classes_[top_2_open_idx]
            top_close_digits = model_close.classes_[top_2_close_idx]
            
            # --- UI PRESENTATION ---
            st.markdown(f"<h2 style='text-align: center; color: #4DA8DA;'>🤖 ML PREDICTION FOR: {next_day_target.upper()} 🤖</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: gray;'>Base Data: {last_day} [{last_open}{last_close}]</p>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<div style='background-color:#1e1e1e; padding:20px; border-radius:10px; border-left: 5px solid #e74c3c;'> "
                    f"<h4 style='color:#e74c3c;'>🔥 ML TRAINED OPEN</h4>"
                    f"<h1 style='font-size: 40px;'>{top_open_digits[0]}, {top_open_digits[1]}</h1>"
                    f"</div>", unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div style='background-color:#1e1e1e; padding:20px; border-radius:10px; border-left: 5px solid #3498db;'> "
                    f"<h4 style='color:#3498db;'>🔥 ML TRAINED CLOSE</h4>"
                    f"<h1 style='font-size: 40px;'>{top_close_digits[0]}, {top_close_digits[1]}</h1>"
                    f"</div>", unsafe_allow_html=True
                )
                
            expected_jodis = [f"{o}{c}" for o in top_open_digits for c in top_close_digits]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background-color:#111; padding:20px; border-radius:10px; text-align: center; border: 1px dashed #2ecc71;'> "
                f"<h3 style='color:#2ecc71;'>💎 ML GENERATED JODIS 💎</h3>"
                f"<h1 style='color:#2ecc71; letter-spacing: 5px; margin: 0;'>{', '.join(expected_jodis)}</h1>"
                f"</div>", unsafe_allow_html=True
            )

    except Exception as general_err:
        st.error(f"❌ Error: {general_err}")
else:
    st.info("👋 Excel file upload karein aur Machine Learning training start karein.")
