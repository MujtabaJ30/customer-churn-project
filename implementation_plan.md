# Deployment Plan — Customer Churn Dashboard with SHAP Explainability

We will build a high-fidelity, interactive **Customer Churn Dashboard** using Streamlit. This dashboard will serve as the deployment vehicle for our machine learning model (`rf_churn_model.pkl`), enabling business stakeholders to analyze customer churn risk, explore key risk drivers globally and individually (via SHAP), and perform "what-if" simulations on customer profiles.

---

## User Review Required

> [!IMPORTANT]
> **New Dependencies Needed**
> We will need to install `streamlit` to build the web application interface. We will add `streamlit` to `requirements.txt` and install it in the virtual environment.

> [!NOTE]
> **Saved SHAP Explainer**
> To avoid recalculating SHAP values dynamically in the app (which is slow), we will export a pre-trained `shap.Explainer` or cache it using joblib/pickle.

---

## Proposed Changes

We will create a structured directory structure for the app, placing the source code under a new `app/` folder to separate production code from exploratory notebooks.

### 1. Web Application Component

#### [NEW] [app.py](file:///e:/Projects/customer-churn-project/app/app.py)
This will be the entry point for the Streamlit dashboard. It will include:
1. **Interactive KPI Metrics**: Overall churn rate, total customers, revenue at risk.
2. **Customer Lookup Tool**: 
   - Retrieve a customer profile by selecting their ID.
   - Display their churn probability and RFM segment.
   - Render a SHAP waterfall/force plot explaining *why* they are predicted to churn.
3. **What-If Churn Simulator**: 
   - Allow sliders and selectors for customer attributes (Contract, Internet Type, Monthly Charges, Tech Support, etc.).
   - Re-run the Random Forest prediction instantly in the background and show the new churn probability.

#### [NEW] [utils.py](file:///e:/Projects/customer-churn-project/app/utils.py)
Helper module containing caching logic:
- `load_data()`: Load and cache the cleaned `master_df.csv` and `rfm_segments.csv`.
- `load_model()`: Load and cache the serialized Random Forest model.
- `get_shap_explainer()`: Load or construct and cache the SHAP explainer object for fast execution.

### 2. Export / Artifact Preparation

#### [MODIFY] [04_churn_prediction.ipynb](file:///e:/Projects/customer-churn-project/notebooks/04_churn_prediction.ipynb)
We will add a clean script cell at the end of the notebook to dump the trained model along with the SHAP explainer object to `models/` directory for the Streamlit app to load.

---

## Verification Plan

### Automated/Local Tests
1. **Dependency Installation**:
   - Propose virtual environment command to install requirements: `pip install streamlit`.
2. **Streamlit App Launch**:
   - Propose command: `streamlit run app/app.py`
3. **Functional Verification via Browser**:
   - Verify that the sidebar sliders dynamically alter prediction probability.
   - Verify that searching for a valid Customer ID (e.g., `8779-QRDMV`) generates the correct SHAP explanation.
