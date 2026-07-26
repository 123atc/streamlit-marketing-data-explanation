import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

THRESHOLD = 0.35
LOG_COLS = ['MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

def prepare_and_predict(df_raw):
    df = df_raw.copy()
    for col in LOG_COLS:
        df[col] = np.log1p(df[col])
    df = df[feature_columns]  # enforce exact training column order
    scaled = scaler.transform(df)
    probs = model.predict_proba(scaled)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds

st.title("Marketing Campaign Response Predictor")

tab1, tab2 = st.tabs(["Single Customer", "Batch Upload (CSV)"])

with tab1:
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Income", 0, 200000, 50000)
        age = st.number_input("Age", 18, 100, 40)
        kidhome = st.number_input("Kids at Home", 0, 3, 0)
        teenhome = st.number_input("Teens at Home", 0, 3, 0)
        recency = st.number_input("Days Since Last Purchase", 0, 100, 30)
        customer_days = st.number_input("Days as Customer", 0, 3000, 500)
        complain = st.selectbox("Filed a Complaint?", [0, 1])
        marital = st.selectbox("Marital Status", 
                    ["Married", "Single", "Together", "Divorced", "Widow"])
        education = st.selectbox("Education", 
                    ["Graduation", "PhD", "Master", "Basic", "2n Cycle"])

    with col2:
        wines = st.number_input("Spent on Wine ($)", 0, 3000, 100)
        fruits = st.number_input("Spent on Fruits ($)", 0, 300, 20)
        meat = st.number_input("Spent on Meat ($)", 0, 2000, 100)
        fish = st.number_input("Spent on Fish ($)", 0, 300, 20)
        sweets = st.number_input("Spent on Sweets ($)", 0, 300, 20)
        gold = st.number_input("Spent on Gold Prods ($)", 0, 400, 20)
        deals = st.number_input("Deal Purchases", 0, 20, 2)
        web = st.number_input("Web Purchases", 0, 20, 3)
        catalog = st.number_input("Catalog Purchases", 0, 20, 1)
        store = st.number_input("Store Purchases", 0, 20, 4)
        webvisits = st.number_input("Web Visits/Month", 0, 20, 5)

    st.markdown("**Past Campaign Responses**")
    c1, c2, c3, c4, c5 = st.columns(5)
    cmp1 = c1.checkbox("Cmp 1")
    cmp2 = c2.checkbox("Cmp 2")
    cmp3 = c3.checkbox("Cmp 3")
    cmp4 = c4.checkbox("Cmp 4")
    cmp5 = c5.checkbox("Cmp 5")

    if st.button("Predict"):
        mnt_total = wines + fruits + meat + fish + sweets + gold
        accepted_overall = sum([cmp1, cmp2, cmp3, cmp4, cmp5])

        row = {
            'Income': income, 'Kidhome': kidhome, 'Teenhome': teenhome,
            'Recency': recency, 'MntWines': wines, 'MntFruits': fruits,
            'MntMeatProducts': meat, 'MntFishProducts': fish,
            'MntSweetProducts': sweets, 'MntGoldProds': gold,
            'NumDealsPurchases': deals, 'NumWebPurchases': web,
            'NumCatalogPurchases': catalog, 'NumStorePurchases': store,
            'NumWebVisitsMonth': webvisits,
            'AcceptedCmp3': int(cmp3), 'AcceptedCmp4': int(cmp4),
            'AcceptedCmp5': int(cmp5), 'AcceptedCmp1': int(cmp1),
            'AcceptedCmp2': int(cmp2), 'Complain': complain,
            'Age': age, 'Customer_Days': customer_days,
            'marital_Divorced': int(marital == "Divorced"),
            'marital_Married': int(marital == "Married"),
            'marital_Single': int(marital == "Single"),
            'marital_Together': int(marital == "Together"),
            'marital_Widow': int(marital == "Widow"),
            'education_2n Cycle': int(education == "2n Cycle"),
            'education_Basic': int(education == "Basic"),
            'education_Graduation': int(education == "Graduation"),
            'education_Master': int(education == "Master"),
            'education_PhD': int(education == "PhD"),
            'MntTotal': mnt_total, 'MntRegularProds': mnt_total,
            'AcceptedCmpOverall': accepted_overall
        }

        input_df = pd.DataFrame([row])
        probs, preds = prepare_and_predict(input_df)

        label = "Likely to Respond ✅" if preds[0] == 1 else "Unlikely to Respond ❌"
        st.success(f"**{label}**")
        st.write(f"Predicted probability: **{probs[0]:.1%}**")

with tab2:
    st.subheader("Upload Customer CSV")
    st.caption("CSV must contain the same raw columns used in training (before log-transform/scaling).")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        probs, preds = prepare_and_predict(batch_df)
        batch_df['Response_Probability'] = probs
        batch_df['Predicted_Response'] = preds
        st.dataframe(batch_df)
        st.download_button("Download Results", 
                            batch_df.to_csv(index=False), 
                            "predictions.csv")