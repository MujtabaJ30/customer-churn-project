import streamlit as st
import matplotlib.pyplot as plt
import shap
from utils import load_data, load_master_data, load_model, load_shap_explainer, load_feature_names, load_encoding_config, encode_customer

st.set_page_config(page_title='Customer Churn Dashboard', layout='wide')

st.title('Customer Churn Dashboard')
st.markdown('Telecom customer retention analysis powered by Random Forest + SHAP')

data = load_data()
master_data = load_master_data()
model = load_model()
explainer = load_shap_explainer()
feature_names = load_feature_names()
encoding_config = load_encoding_config()

total_customers = len(data)
churn_rate = data['Churn Value'].mean()
at_risk = data[data['Churn Probability'] > 0.5]
at_risk_count = len(at_risk)
revenue_at_risk = at_risk['Total Revenue'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Customers', f'{total_customers:,}')
col2.metric('Churn Rate', f'{churn_rate:.1%}')
col3.metric('At-Risk Customers', f'{at_risk_count:,}')
col4.metric('Revenue at Risk', f'${revenue_at_risk:,.0f}')

st.divider()
st.subheader('Customer Lookup')

customer_ids = data['Customer ID'].tolist()
selected_id = st.selectbox('Select a Customer ID', customer_ids)

customer = data[data['Customer ID'] == selected_id].iloc[0]
prob = customer['Churn Probability']
segment = customer['RFM_Segment']
churned = customer['Churn Value']
revenue = customer['Total Revenue']

col_a, col_b = st.columns([1, 2])
with col_a:
    st.markdown(f'### Customer {selected_id}')
    risk_label = 'High Risk' if prob > 0.5 else 'Low Risk'
    risk_color = 'red' if prob > 0.5 else 'green'
    st.markdown(f'**Churn Probability:**  <span style="color:{risk_color}; font-size:1.3rem">{prob:.0%}</span> ({risk_label})', unsafe_allow_html=True)
    st.markdown(f'**RFM Segment:** {segment}')
    st.markdown(f'**Status:** {"Churned" if churned else "Retained"}')
    st.markdown(f'**Total Revenue:** ${revenue:,.0f}')

with col_b:
    raw_customer = master_data[master_data['Customer ID'] == selected_id]
    if not raw_customer.empty:
        customer_vector = encode_customer(raw_customer.iloc[0], encoding_config, feature_names)
        shap_values = explainer.shap_values(customer_vector)
        shap_class1 = shap_values[:, :, 1]
        explanation = shap.Explanation(
            values=shap_class1[0],
            base_values=explainer.expected_value[1],
            data=customer_vector.iloc[0],
            feature_names=feature_names
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(explanation, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption('Top factors driving churn risk for this customer')

st.divider()
st.subheader('What-If Churn Simulator')
st.markdown('Modify any attribute and see how churn probability changes in real time.')

with st.expander('Open Simulator', expanded=True):
    raw_customer_row = master_data[master_data['Customer ID'] == selected_id].iloc[0].copy()

    with st.form('what_if_form'):
        col_demo, col_acct, col_serv = st.columns(3)

        with col_demo:
            st.markdown('**Demographics**')
            wi_gender = st.selectbox('Gender', ['Male', 'Female'],
                index=0 if raw_customer_row['Gender'] == 'Male' else 1)
            wi_age = st.slider('Age', 18, 100, int(raw_customer_row['Age']))
            wi_under_30 = st.selectbox('Under 30', ['Yes', 'No'],
                index=0 if raw_customer_row['Under 30'] == 'Yes' else 1)
            wi_senior = st.selectbox('Senior Citizen', ['Yes', 'No'],
                index=0 if raw_customer_row['Senior Citizen'] == 'Yes' else 1)
            wi_married = st.selectbox('Married', ['Yes', 'No'],
                index=0 if raw_customer_row['Married'] == 'Yes' else 1)
            wi_dependents = st.selectbox('Dependents', ['Yes', 'No'],
                index=0 if raw_customer_row['Dependents'] == 'Yes' else 1)
            wi_num_dep = st.slider('Number of Dependents', 0, 10,
                int(raw_customer_row['Number of Dependents']))
            wi_referred = st.selectbox('Referred a Friend', ['Yes', 'No'],
                index=0 if raw_customer_row['Referred a Friend'] == 'Yes' else 1)
            wi_referrals = st.slider('Number of Referrals', 0, 20,
                int(raw_customer_row['Number of Referrals']))

        with col_acct:
            st.markdown('**Account**')
            wi_tenure = st.slider('Tenure (Months)', 0, 72,
                int(raw_customer_row['Tenure in Months']))
            wi_offer = st.selectbox('Offer',
                ['No Offer', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'],
                index=['No Offer', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E']
                    .index(raw_customer_row['Offer']))
            wi_contract = st.selectbox('Contract',
                ['Month-to-Month', 'One Year', 'Two Year'],
                index=['Month-to-Month', 'One Year', 'Two Year']
                    .index(raw_customer_row['Contract']))
            wi_paperless = st.selectbox('Paperless Billing', ['Yes', 'No'],
                index=0 if raw_customer_row['Paperless Billing'] == 'Yes' else 1)
            wi_payment = st.selectbox('Payment Method',
                ['Bank Withdrawal', 'Credit Card', 'Mailed Check'],
                index=['Bank Withdrawal', 'Credit Card', 'Mailed Check']
                    .index(raw_customer_row['Payment Method']))
            wi_monthly = st.slider('Monthly Charge ($)', 0.0, 200.0,
                float(raw_customer_row['Monthly Charge']))
            wi_total_charges = st.slider('Total Charges ($)', 0.0, 15000.0,
                float(raw_customer_row['Total Charges']))
            wi_refunds = st.slider('Total Refunds ($)', 0.0, 3000.0,
                float(raw_customer_row['Total Refunds']))
            wi_total_revenue = st.slider('Total Revenue ($)', 0.0, 25000.0,
                float(raw_customer_row['Total Revenue']))

        with col_serv:
            st.markdown('**Services**')
            wi_phone = st.selectbox('Phone Service', ['Yes', 'No'],
                index=0 if raw_customer_row['Phone Service'] == 'Yes' else 1)
            wi_avg_ld = st.slider('Avg Monthly Long Distance ($)', 0.0, 200.0,
                float(raw_customer_row['Avg Monthly Long Distance Charges']))
            wi_multiline = st.selectbox('Multiple Lines', ['Yes', 'No'],
                index=0 if raw_customer_row['Multiple Lines'] == 'Yes' else 1)
            wi_internet = st.selectbox('Internet Service', ['Yes', 'No'],
                index=0 if raw_customer_row['Internet Service'] == 'Yes' else 1)
            wi_internet_type = st.selectbox('Internet Type',
                ['DSL', 'Fiber Optic', 'Cable', 'No Internet'],
                index=['DSL', 'Fiber Optic', 'Cable', 'No Internet']
                    .index(raw_customer_row['Internet Type']))
            wi_avg_gb = st.slider('Avg Monthly GB Download', 0, 1000,
                int(raw_customer_row['Avg Monthly GB Download']))
            wi_online_sec = st.selectbox('Online Security', ['Yes', 'No'],
                index=0 if raw_customer_row['Online Security'] == 'Yes' else 1)
            wi_online_backup = st.selectbox('Online Backup', ['Yes', 'No'],
                index=0 if raw_customer_row['Online Backup'] == 'Yes' else 1)
            wi_device = st.selectbox('Device Protection Plan', ['Yes', 'No'],
                index=0 if raw_customer_row['Device Protection Plan'] == 'Yes' else 1)
            wi_support = st.selectbox('Premium Tech Support', ['Yes', 'No'],
                index=0 if raw_customer_row['Premium Tech Support'] == 'Yes' else 1)

        col_engage, col_stream, col_extra_col = st.columns(3)

        with col_engage:
            st.markdown('**Engagement**')
            wi_satisfaction = st.slider('Satisfaction Score', 1, 5,
                int(raw_customer_row['Satisfaction Score']))
            wi_extra_data = st.slider('Total Extra Data Charges ($)', 0.0, 3000.0,
                float(raw_customer_row['Total Extra Data Charges']))
            wi_total_ld = st.slider('Total Long Distance Charges ($)', 0.0, 5000.0,
                float(raw_customer_row['Total Long Distance Charges']))

        with col_stream:
            st.markdown('**Streaming & Add-ons**')
            wi_tv = st.selectbox('Streaming TV', ['Yes', 'No'],
                index=0 if raw_customer_row['Streaming TV'] == 'Yes' else 1)
            wi_movies = st.selectbox('Streaming Movies', ['Yes', 'No'],
                index=0 if raw_customer_row['Streaming Movies'] == 'Yes' else 1)
            wi_music = st.selectbox('Streaming Music', ['Yes', 'No'],
                index=0 if raw_customer_row['Streaming Music'] == 'Yes' else 1)
            wi_unlimited = st.selectbox('Unlimited Data', ['Yes', 'No'],
                index=0 if raw_customer_row['Unlimited Data'] == 'Yes' else 1)

        with col_extra_col:
            st.markdown('')
            wi_offer_display = st.markdown('_All other values carried over from customer profile._')

        submitted = st.form_submit_button('Run Prediction', type='primary',
            use_container_width=True)

    if submitted:
        raw_customer_row['Gender'] = wi_gender
        raw_customer_row['Age'] = wi_age
        raw_customer_row['Under 30'] = wi_under_30
        raw_customer_row['Senior Citizen'] = wi_senior
        raw_customer_row['Married'] = wi_married
        raw_customer_row['Dependents'] = wi_dependents
        raw_customer_row['Number of Dependents'] = wi_num_dep
        raw_customer_row['Referred a Friend'] = wi_referred
        raw_customer_row['Number of Referrals'] = wi_referrals
        raw_customer_row['Tenure in Months'] = wi_tenure
        raw_customer_row['Offer'] = wi_offer
        raw_customer_row['Phone Service'] = wi_phone
        raw_customer_row['Avg Monthly Long Distance Charges'] = wi_avg_ld
        raw_customer_row['Multiple Lines'] = wi_multiline
        raw_customer_row['Internet Service'] = wi_internet
        raw_customer_row['Internet Type'] = wi_internet_type
        raw_customer_row['Avg Monthly GB Download'] = wi_avg_gb
        raw_customer_row['Online Security'] = wi_online_sec
        raw_customer_row['Online Backup'] = wi_online_backup
        raw_customer_row['Device Protection Plan'] = wi_device
        raw_customer_row['Premium Tech Support'] = wi_support
        raw_customer_row['Streaming TV'] = wi_tv
        raw_customer_row['Streaming Movies'] = wi_movies
        raw_customer_row['Streaming Music'] = wi_music
        raw_customer_row['Unlimited Data'] = wi_unlimited
        raw_customer_row['Contract'] = wi_contract
        raw_customer_row['Paperless Billing'] = wi_paperless
        raw_customer_row['Payment Method'] = wi_payment
        raw_customer_row['Monthly Charge'] = wi_monthly
        raw_customer_row['Total Charges'] = wi_total_charges
        raw_customer_row['Total Refunds'] = wi_refunds
        raw_customer_row['Total Extra Data Charges'] = wi_extra_data
        raw_customer_row['Total Long Distance Charges'] = wi_total_ld
        raw_customer_row['Total Revenue'] = wi_total_revenue
        raw_customer_row['Satisfaction Score'] = wi_satisfaction

        encoded_sim = encode_customer(raw_customer_row, encoding_config, feature_names)
        new_prob = model.predict_proba(encoded_sim)[0, 1]
        prob_change = new_prob - prob

        col_res, col_shap_sim = st.columns([1, 2])
        with col_res:
            st.markdown('### Simulation Result')
            st.metric('Churn Probability', f'{new_prob:.1%}',
                delta=f'{prob_change:+.1%}',
                delta_color='inverse')
            if new_prob > 0.5:
                st.error('High Risk — retention action recommended')
            else:
                st.success('Low Risk')

        with col_shap_sim:
            shap_sim = explainer.shap_values(encoded_sim)
            shap_sim_c1 = shap_sim[:, :, 1]
            explanation_sim = shap.Explanation(
                values=shap_sim_c1[0],
                base_values=explainer.expected_value[1],
                data=encoded_sim.iloc[0],
                feature_names=feature_names
            )
            fig_sim, ax_sim = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(explanation_sim, max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig_sim)
            st.caption('Updated SHAP explanation based on modified attributes')
