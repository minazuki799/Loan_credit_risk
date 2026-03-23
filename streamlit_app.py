import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# page config
st.set_page_config(
    page_title="Loan Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# model loading
with open("credit_risk_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data['model']
preprocessor = model_data['preprocessor']
threshold = model_data['threshold']

# Sidebar navigation
st.sidebar.title("🏦 Loan Credit Risk Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Predict", "Scenario Analysis", "ℹ️ About"]
)

  #-------- input details ----------
    # 'person_income', 'person_home_ownership', 'person_emp_length',
    #        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
    #        'loan_status', 'loan_percent_income', 'cb_person_default_on_file',
    #        'credit_start_year']

# ***********************************************

def show_prediction_page():
    st.title("🔍 Prediction Page")
        
    # expander with glossary
    with st.expander("ℹ️ What do these fields mean?"):
        st.write("""
        **💰 Financial Details**
        
        - **Annual Income** — The applicant's total annual income in dollars. Higher income generally indicates better repayment ability.
        
        - **Loan Amount** — The total amount of money being requested by the applicant.
        
        - **Interest Rate (%)** — The annual interest rate assigned to the loan. Higher rates are typically assigned to riskier applicants.
        
        - **Loan % of Income** — The ratio of the loan amount to the applicant's annual income. A higher ratio means a greater financial burden relative to earnings.
        
        **👤 Personal Details**
        
        - **Employment Length** — Number of years the applicant has been employed. Longer employment suggests greater financial stability.
        
        - **Credit Entry Age** — The age of the applicant when the applicant's credit history began. A longer credit history generally indicates a more reliable borrower.
        
        - **Loan Grade** — A credit rating assigned to the loan based on the applicant's credit profile:
            - *A* — Excellent — lowest risk
            - *B* — Very good
            - *C* — Good
            - *D* — Fair
            - *E* — Poor
            - *F* — Very poor
            - *G* — Highest risk
        
        - **Home Ownership** — The applicant's current housing status:
            - *RENT* — Currently renting
            - *OWN* — Owns home outright
            - *MORTGAGE* — Paying off a mortgage
            - *OTHER* — Other living arrangement
        
        - **Loan Intent** — The stated purpose for requesting the loan:
            - *PERSONAL* — General personal use
            - *EDUCATION* — Funding education or tuition
            - *MEDICAL* — Covering medical expenses
            - *VENTURE* — Starting or funding a business
            - *HOMEIMPROVEMENT* — Home renovation or repairs
            - *DEBTCONSOLIDATION* — Combining existing debts into one
        
        - **Previous Default** — Whether the applicant has a previously recorded loan default:
            - *Y* — Yes, a previous default exists — higher risk
            - *N* — No previous default on record — lower risk
        """)

    #col for input
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("💰 Financial Details")
        person_income = st.number_input("Annual Income", min_value=0, help= "The applicant's total annual income in dollars")
        loan_amnt = st.number_input("Loan Amount", min_value=0, help="Total amount of money being requested")
        loan_int_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0, help="The interest rate assigned to the loan")
        loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.1,help="Ratio of loan amount to annual income — higher means more financial burden")

    with col2:
        st.subheader("👤 Personal Details")
        person_emp_length = st.slider("Employment Length", 0, 50, 5,help="Number of years the applicant has been employed")
        credit_start_year = st.number_input("Credit Entry Age", 14, 100, 18,help="The age of the applicant when the applicant's credit history began.)
        loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"],help="Credit grade assigned to the loan — A is best, G is highest risk")
        person_home_ownership = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE","OTHER"],help="The applicant's current home ownership status")
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"],help="The purpose for which the loan is being requested")
        cb_person_default_on_file = st.radio("Previous Default", ["Y","N"], horizontal=True,help="Whether the applicant has a previous loan default recorded — Y = Yes, N = No")
    st.divider()

    customer = {
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "credit_start_year": credit_start_year   
    }

    # predict section
    
    # helper functions
    def feature_imp():
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).tail(10).sort_values("Importance")

        # Clean up feature names — remove prefixes like "num__", "ord__", "one__"
        importance_df["Feature"] = importance_df["Feature"].str.replace(r"^(num__|ord__|one__)", "", regex=True)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="🔍 Top 10 Most Influential Features",
            color="Importance",
            color_continuous_scale="Blues",
            text=importance_df["Importance"].apply(lambda x: f"{x:.4f}")
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
            title_font=dict(size=16),
            coloraxis_showscale=False,
            xaxis_title="Importance Score",
            yaxis_title="",
            margin=dict(l=10, r=10, t=50, b=10),
            height=450
        )

        fig.update_traces(
            textposition="outside",
            marker_line_color="steelblue",
            marker_line_width=1
        )

        st.plotly_chart(fig, use_container_width=True)

    # summary table
    def summary():
        st.subheader("📋 User Summary")
        summary = pd.DataFrame({
            "Feature": [
                "Annual Income", "Loan Amount", "Interest Rate",
                "Loan % of Income", "Employment Length", "Loan Grade",
                "Home Ownership", "Loan Intent", "Previous Default",
                "Credit Entry Age"
            ],
            "Value": [
                f"${person_income:,.0f}", f"${loan_amnt:,.0f}",
                f"{loan_int_rate:.1f}%", f"{loan_percent_income:.2f}",
                f"{person_emp_length} years", loan_grade,
                person_home_ownership, loan_intent,
                cb_person_default_on_file, credit_start_year
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if st.button('Predict'):
        X = preprocessor.transform(pd.DataFrame([customer]))
        y_prob = model.predict_proba(X)[0,1]
        y_pred = int(y_prob >= threshold)

        st.divider()
        st.subheader("📊 Prediction Result")

        # Three metrics side by side
        col1, col2, col3 = st.columns(3)

        col1.metric("Default Probability", f"{y_prob:.2%}")
        col2.metric("Decision", "Default" if y_pred == 1 else "No Default")
        col3.metric("Threshold Used", f"{threshold:.2f}")

        # Probability bar
        st.write("**Risk Level:**")
        st.progress(float(y_prob))

        # Colour coded result
        if y_prob < threshold:
            st.success("✅ Low Risk — Loan strongly recommended.")
        else:
            st.error("🚨 High Risk — Loan not recommended.")
        
        #visualizations
        feature_imp()
        summary()

        # Save to session state for scenario analysis
        st.session_state.y_prob = y_prob
        st.session_state.y_pred = y_pred
        st.session_state.customer = customer

if page == "🔍 Predict":
    show_prediction_page()

# *********************************************

# about page 
def show_about_page():
    st.title("ℹ️ About This App")
    st.subheader("📌 Project Overview")
    st.write("""
    This application uses a machine learning model to predict the likelihood 
    of a loan applicant defaulting on their loan. It is designed to assist 
    financial institutions in making faster, more consistent, and data-driven 
    lending decisions.
    """)

    st.divider()

    st.subheader("📊 Dataset")
    st.write("""
    The model was trained on a credit loan dataset containing over 30,000 
    applicant records with 10 features(9 selected, 1 engineered)
    """)

    st.divider()

    st.subheader("🤖 Model Details")

    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - **Algorithm:** XGBoost Classifier
        - **Roc_auc_score:** 0.87
        - **Precision:** 80%
        - **Recall:** 80%
        - **F1 Score:** 80%
        - **Threshold:** 0.31
        """)
    with col2:
        st.write("""
        - **Training size:** 70%
        - **Test size:** 30%
        - **Cross validation:** 5-fold
        - **Optimisation:** GridSearchCV
        - **Imbalance handling:** class_weight=balanced
        """)

    st.divider()

    st.subheader("🛠️ Tech Stack")
    col1, col2, col3 = st.columns(3)
    col1.info("**Data**\n\nPandas\nNumPy\nScikit-learn\nswiftmltoolz")
    col2.info("**Model**\n\nXGBoost\nGridSearchCV")
    col3.info("**App**\n\nStreamlit\nPlotly")

    st.divider()

    st.subheader("👤 About the Developer")
    st.write("""
    Built by **Victor Okosun** — an aspiring ml engineer 
    with self-directed expertise in Python, machine learning, and AI.
    
    - 🔗 GitHub: [github.com/minazuki799](https://github.com/minazuki799)
    - 💼 LinkedIn: [linkedin.com/in/victor-okosun](https://linkedin.com/in/victor-okosun)
    """)

if page == "ℹ️ About":
    show_about_page()

# ***********************************************

# scenario analysis

def show_scenario_analysis_page():
    st.title("Scenario Analysis")
    
    # Check if user has made a prediction first
    if "y_prob" not in st.session_state or st.session_state.y_prob is None:
        st.warning("⚠️ Please make a prediction on the Predict page first.")
        st.stop()
    
    # Baseline from original prediction
    baseline_prob = st.session_state.y_prob
    baseline_customer = st.session_state.customer
    
    st.info(f"🔵 Baseline prediction: **{baseline_prob:.2%}** probability of default. Adjust any feature below to see how it affects the risk.")
    
    st.divider()
    st.subheader("🎛️ Adjust a Feature")
    
    # Let user pick which feature to adjust
    feature_to_change = st.selectbox(
        "Select feature to adjust",
        [
            "Annual Income",
            "Loan Amount", 
            "Interest Rate (%)",
            "Loan % of Income",
            "Employment Length",
            "Credit Entry Age",
            "Loan Grade",
            "Home Ownership",
            "Loan Intent",
            "Previous Default"
        ]
    )
    
    # Show only the relevant input for selected feature
    customer_wi = baseline_customer.copy()
    
    if feature_to_change == "Annual Income":
        new_val = st.slider("Annual Income", 0, 200000, 
                           int(baseline_customer["person_income"]), step=1000)
        customer_wi["person_income"] = new_val

    elif feature_to_change == "Loan Amount":
        new_val = st.slider("Loan Amount", 0, 50000, 
                           int(baseline_customer["loan_amnt"]), step=500)
        customer_wi["loan_amnt"] = new_val

    elif feature_to_change == "Interest Rate (%)":
        new_val = st.slider("Interest Rate (%)", 0.0, 30.0, 
                           float(baseline_customer["loan_int_rate"]), step=0.1)
        customer_wi["loan_int_rate"] = new_val

    elif feature_to_change == "Loan % of Income":
        new_val = st.slider("Loan % of Income", 0.0, 1.0, 
                           float(baseline_customer["loan_percent_income"]), step=0.01)
        customer_wi["loan_percent_income"] = new_val

    elif feature_to_change == "Employment Length":
        new_val = st.slider("Employment Length", 0, 50, 
                           int(baseline_customer["person_emp_length"]))
        customer_wi["person_emp_length"] = new_val

    elif feature_to_change == "Credit Entry Age":
        new_val = st.number_input("Credit Entry Age", 14, 100, 
                                  int(baseline_customer["credit_start_year"]))
        customer_wi["credit_start_year"] = new_val

    elif feature_to_change == "Loan Grade":
        new_val = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"],
                               index=["A","B","C","D","E","F","G"].index(baseline_customer["loan_grade"]))
        customer_wi["loan_grade"] = new_val

    elif feature_to_change == "Home Ownership":
        options = ["RENT","OWN","MORTGAGE","OTHER"]
        new_val = st.selectbox("Home Ownership", options,
                               index=options.index(baseline_customer["person_home_ownership"]))
        customer_wi["person_home_ownership"] = new_val

    elif feature_to_change == "Loan Intent":
        options = ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"]
        new_val = st.selectbox("Loan Intent", options,
                               index=options.index(baseline_customer["loan_intent"]))
        customer_wi["loan_intent"] = new_val

    elif feature_to_change == "Previous Default":
        new_val = st.radio("Previous Default", ["Y","N"], horizontal=True,
                           index=["Y","N"].index(baseline_customer["cb_person_default_on_file"]))
        customer_wi["cb_person_default_on_file"] = new_val

    # New prediction
    X_wi = preprocessor.transform(pd.DataFrame([customer_wi]))
    y_prob_wi = model.predict_proba(X_wi)[0, 1]
    y_pred_wi = int(y_prob_wi >= threshold)

    # Compare baseline vs new
    delta = y_prob_wi - baseline_prob

    st.divider()
    st.subheader("📊 Scenario Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Probability", f"{baseline_prob:.2%}")
    col2.metric("New Probability", f"{y_prob_wi:.2%}",
                delta=f"{delta:+.2%}",
                delta_color="inverse")
    col3.metric("Change", f"{delta:+.2%}",
                delta_color="inverse")

    st.progress(float(y_prob_wi))

    if y_pred_wi == 1:
        st.error("🚨 High Risk — Loan not recommended.")
    else:
        st.success("✅ Low Risk — Loan recommended.")

    # Explanation
    if delta > 0:
        st.warning(f"⬆️ Changing **{feature_to_change}** increased default risk by **{delta:.2%}**")
    elif delta < 0:
        st.info(f"⬇️ Changing **{feature_to_change}** decreased default risk by **{abs(delta):.2%}**")
    else:
        st.info("➡️ This change had no effect on the prediction.")

    
if page == "Scenario Analysis":
    show_scenario_analysis_page()
