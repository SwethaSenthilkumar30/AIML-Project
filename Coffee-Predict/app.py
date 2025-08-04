import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #e8e8e8;
        font-weight: 300;
    }
    
    /* Input section styling */
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .input-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Prediction result styling */
    .prediction-container {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        margin: 0;
    }
    
    /* Decision path styling */
    .decision-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .decision-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #764ba2;
    }
    
    .decision-step {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-radius: 10px;
        font-weight: 500;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #fff;
    }
    
    .leaf-node {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        border: 3px solid #fff;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Label styling */
    .stNumberInput > label {
        font-weight: 600;
        color: #555;
        font-size: 1.1rem;
    }
    
    /* Coffee quality class styling */
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem;
    }
    
    .quality-0 { background: #ff6b6b; color: white; }
    .quality-1 { background: #feca57; color: white; }
    .quality-2 { background: #48dbfb; color: white; }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Step 1: Define the data (copied from your notebook) ---
X = [
    [6.2, 3.3, 6.6, 2.2],
    [5.1, 3.5, 1.4, 0.2],
    [5.7, 2.8, 4.5, 1.3],
    [6.3, 2.9, 5.6, 1.8],
    [4.9, 3.0, 1.4, 0.2],
    [5.0, 2.3, 3.3, 1.0],
    [6.7, 3.0, 5.2, 2.3],
    [5.4, 3.7, 1.5, 0.2],
    [5.6, 2.7, 4.2, 1.3],
    [6.5, 3.0, 5.8, 2.2]
]

y = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

feature_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
quality_labels = {0: "Low Quality", 1: "Medium Quality", 2: "High Quality"}

# --- Step 2: Train the model ---
model = DecisionTreeClassifier()
model.fit(X, y)

# --- Step 3: Streamlit UI ---
st.markdown("""
<div class="title-container">
    <h1 class="main-title">🌱 Coffee Quality Predictor</h1>
    <p class="subtitle">AI-powered decision tree analysis for coffee quality assessment</p>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="input-header">📊 Enter Coffee Bean Measurements</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

user_input = []
with col1:
    for i in range(0, 2):
        name = feature_names[i]
        val = st.number_input(
            f"☕ {name}", 
            value=float(np.mean([x[i] for x in X])), 
            step=0.1,
            key=f"input_{i}"
        )
        user_input.append(val)

with col2:
    for i in range(2, 4):
        name = feature_names[i]
        val = st.number_input(
            f"🌿 {name}", 
            value=float(np.mean([x[i] for x in X])), 
            step=0.1,
            key=f"input_{i}"
        )
        user_input.append(val)

st.markdown('</div>', unsafe_allow_html=True)

input_sample = np.array(user_input).reshape(1, -1)

if st.button("🔮 Predict Coffee Quality"):
    prediction = model.predict(input_sample)[0]
    quality_name = quality_labels[prediction]
    
    # Prediction result
    st.markdown(f"""
    <div class="prediction-container">
        <p class="prediction-text">
            Predicted Quality: <span class="quality-badge quality-{prediction}">{quality_name} (Class {prediction})</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Decision path
    st.markdown('<div class="decision-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="decision-header">🧭 Decision Tree Path Analysis</h2>', unsafe_allow_html=True)

    tree_ = model.tree_
    node_indicator = model.decision_path(input_sample)
    leave_id = model.apply(input_sample)

    for node_id in node_indicator.indices:
        if node_id == leave_id[0]:
            st.markdown(f"""
            <div class="leaf-node">
                🎯 Final Decision: Reached leaf node {node_id}
            </div>
            """, unsafe_allow_html=True)
            break

        feature = feature_names[tree_.feature[node_id]]
        threshold = tree_.threshold[node_id]
        value = input_sample[0, tree_.feature[node_id]]

        if value <= threshold:
            direction = "left"
            symbol = "🔸"
            condition = f"{feature} ≤ {threshold:.2f}"
        else:
            direction = "right"
            symbol = "🔹"
            condition = f"{feature} > {threshold:.2f}"
        
        st.markdown(f"""
        <div class="decision-step">
            {symbol} {condition} → go {direction} (value: {value:.2f})
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Add information section
st.markdown("""
<div class="input-section" style="margin-top: 2rem;">
    <h3 style="color: #333; text-align: center; margin-bottom: 1rem;">📖 About This Model</h3>
    <p style="color: #666; text-align: center; line-height: 1.6;">
        This decision tree classifier analyzes coffee bean characteristics to predict quality levels. 
        The model considers sepal and petal measurements to make predictions about coffee quality grades.
    </p>
    <div style="text-align: center; margin-top: 1rem;">
        <span class="quality-badge quality-0">Class 0: Low Quality</span>
        <span class="quality-badge quality-1">Class 1: Medium Quality</span>
        <span class="quality-badge quality-2">Class 2: High Quality</span>
    </div>
</div>
""", unsafe_allow_html=True)