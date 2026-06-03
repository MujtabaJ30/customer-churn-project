# Customer Segmentation & Retention Analysis — Master Project Context

## Goal

Build a FAANG-level end-to-end Data Science portfolio project that demonstrates the complete lifecycle from raw data ingestion to business decision-making and deployment.

The objective is not merely to build a churn model. The objective is to demonstrate the workflow expected from a professional Data Scientist:

* Data acquisition
* Data cleaning
* Data modelling
* Customer segmentation
* Business interpretation
* Customer Lifetime Value analysis
* Model explainability
* Deployment
* Portfolio presentation

Every decision should maximize interview value, portfolio quality, and real-world business relevance.

The project should ultimately be strong enough to discuss confidently in Data Scientist, ML Engineer, Analytics, and related interviews.

---

## User Profile

User is intermediate in Python and Data Science concepts but has limited experience building complete projects independently.

The user is not acting as project architect.

Claude is responsible for:

* Choosing the best project direction
* Choosing the best technical approaches
* Deciding roadmap priorities
* Explaining decisions
* Teaching the reasoning behind every step

The user executes the work.

The goal is not just completion.

The goal is understanding.

The user must be able to explain every major decision during interviews.

---

## Working Style — NON-NEGOTIABLE

Follow these rules strictly.

### 1. One Small Step At A Time

Never jump multiple steps ahead.

Never provide large implementation dumps.

Only provide the next logical step.

The user will say "next" when ready.

---

### 2. Explain Before Coding

Before every implementation step explain:

* Why we are doing it
* What business problem it solves
* Why it matters
* Alternative approaches
* Why the chosen approach is preferred

Reasoning always comes before code.

---

### 3. Build Thinking, Not Copy-Pasting

The user must understand:

* Why the decision was made
* What alternatives existed
* What tradeoffs were accepted

Every major decision should improve interview readiness.

---

### 4. Interview Focus

For important decisions include:

* Possible interview questions
* Strong answers
* Common mistakes
* Decision notes

---

### 5. Professional Standards

Prefer approaches that improve:

* Reproducibility
* Explainability
* Deployment readiness
* Business impact
* Portfolio quality

Do not optimize for speed.

Optimize for project quality.

---

### 6. Context Management

This document is the project source of truth.

Read it first before continuing work.

Do not ask the user to repeat information already documented here.

---

## Tech Stack / Environment

### Operating System

Windows

### IDE

VS Code

### Environment

Python virtual environment (venv)

### Python Version

3.11.9

### Installed Packages

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* jupyter
* ipykernel
* openpyxl
* joblib

---

## Project Structure

customer-churn-project/

data/

raw/

Telco_customer_churn.xlsx

Telco_customer_churn_demographics.xlsx

Telco_customer_churn_location.xlsx

Telco_customer_churn_population.xlsx

Telco_customer_churn_services.xlsx

Telco_customer_churn_status.xlsx

processed/

master_df.csv

rfm_segments.csv

models/

rf_churn_model.pkl

notebooks/

01_data_exploration.ipynb

02_eda.ipynb

03_rfm_segmentation.ipynb

04_churn_prediction.ipynb

05_cltv_analysis.ipynb

confusion_matrix.png

feature_importance.png

rfm_churn_combined.png

rfm_churn_count.png

rfm_churn_rate.png

venv/

requirements.txt

---

## Dataset

IBM Telco Customer Churn Dataset

Sources:

https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113

https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

---

## Dataset Summary

7043 customers

Five relational tables merged through:

Customer ID

Target variable:

Churn Value

Binary classification:

0 = retained

1 = churned

Overall churn rate:

26.5%

---

# Progress Completed

## Notebook 01 — Data Exploration

Completed.

Activities:

* Inspected all source files
* Understood schema
* Identified join key
* Reviewed column meanings
* Assessed data quality

---

## Notebook 02 — EDA

Completed.

Activities:

* Univariate analysis
* Churn distribution analysis
* Feature exploration
* Revenue exploration
* Tenure exploration
* Contract analysis
* Business pattern discovery

---

## Notebook 03 — RFM Segmentation

Completed.

Activities:

* Created telecom-specific RFM framework
* Scored customers
* Built segments
* Measured churn by segment
* Measured revenue by segment

---

## Notebook 04 — Churn Prediction

Completed.

Activities:

* Feature engineering
* Encoding
* Model comparison
* Model evaluation
* Feature importance
* Confusion matrix
* Churn probability generation
* Business action matrix

---

## Notebook 05 — CLTV Analysis

Currently active.

Not yet completed.

---

# Major Decisions Made

## Data Merge Decision

Merged all five source tables.

Reason:

ML requires one row per customer.

Real-world data often exists in relational tables.

Alternative:

SQL joins before ingestion.

---

## Dropped Count And Quarter

Reason:

Count provided no analytical value.

Quarter had no variance.

Both would create merge conflicts and noise.

---

## Null Handling

Did NOT drop rows.

Reason:

Most nulls represented "not applicable".

Examples:

* No Offer
* No Internet
* No Churn

Dropping rows would remove approximately 73% of dataset.

Filled with meaningful categories instead.

---

## Processed Dataset Strategy

Saved cleaned dataset as:

master_df.csv

Reason:

Avoid repeating merge and cleaning work.

Standard production workflow.

Raw data remains untouched.

---

## EDA Before Modelling

Reason:

Understand feature behaviour before training.

Identify business drivers.

Improve explainability.

Support stakeholder communication.

---

## RFM Framework Design

Traditional RFM unavailable.

Adapted telecom-specific version:

Recency:

Tenure in Months

Frequency:

Number of Referrals

Monetary:

Total Revenue

Reason:

Most suitable behavioural proxies available.

Avoided using IBM CLTV field.

---

## Quartile Scoring Strategy

Used qcut where possible.

Reason:

Equal population distribution.

More robust than manual ranges.

Alternative:

pd.cut

Rejected because data was not evenly distributed.

---

## Referral Frequency Scoring

qcut failed.

Reason:

Excessive duplicate values.

Large number of zero referrals.

Solution:

Manual threshold scoring.

Accepted because distribution is highly skewed.

---

## RFM Score

Final formula:

R + F + M

Range:

3–12

---

## RFM Segment Performance

Champions

Customers: 1395

Churn Rate: 6.5%

Average Revenue: $6570

---

Loyal Customers

Customers: 2384

Churn Rate: 18.4%

Average Revenue: $4009

---

At Risk

Customers: 1527

Churn Rate: 30.1%

Average Revenue: $1405

---

Lost

Customers: 1737

Churn Rate: 50.6%

Average Revenue: $288

---

## Business Retention Decision

Champions

Protect.

Do not aggressively target.

---

Loyal Customers

Monitor.

Maintain engagement.

---

At Risk

Primary retention target.

High churn risk.

Still economically valuable.

---

Lost

Retention ROI poor.

Do not prioritize.

---

# Feature Engineering Decisions

Dropped:

Customer ID

Reason:

Identifier only.

No predictive signal.

---

Dropped leakage columns:

* Churn Label
* Customer Status
* Churn Category
* Churn Reason
* Churn Score

Reason:

Contain post-churn information.

Would cause data leakage.

---

Dropped:

CLTV

Reason:

IBM-generated metric.

Unknown methodology.

Potential leakage.

---

Dropped derived RFM columns.

Reason:

Avoid redundancy and multicollinearity.

---

Dropped low-value geographic columns:

* Zip Code
* Latitude
* Longitude
* City

Reason:

Low generalization value.

---

# Encoding Strategy

Binary columns:

Label Encoding

0/1

Reason:

No information loss.

---

Multi-category columns:

One-Hot Encoding

Reason:

No natural ordering.

Avoid false ordinal relationships.

---

Boolean outputs converted to integers.

Reason:

Consistency.

---

# Modelling Decisions

Train-Test Split:

80/20

stratify=y

Reason:

Preserve churn ratio.

---

Class Imbalance Strategy

class_weight='balanced'

Reason:

74/26 imbalance.

---

Models Evaluated

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

---

## Model Results

Logistic Regression

Recall: 0.93

Precision: 0.85

F1: 0.89

---

Random Forest

Recall: 0.91

Precision: 0.93

F1: 0.92

---

Gradient Boosting

Recall: 0.89

Precision: 0.97

F1: 0.93

---

## Final Model Selection

Random Forest

Reason:

Best balance between:

* Recall
* Precision
* Business usability

Logistic Regression produced too many false alarms.

Gradient Boosting missed too many churners.

---

## Churn Probability Analysis

Average probability:

0.295

Actual churn rate:

0.265

Slight overestimation.

Acceptable.

Probability distribution showed strong customer separation.

Desirable for retention targeting.

---

# RFM + Churn Combined Analysis

Segment Summary:

Lost

1737 customers

Average churn probability:

0.54

Actual churn:

879

Churn rate:

50.6%

---

At Risk

1527 customers

Average churn probability:

0.34

Actual churn:

460

Churn rate:

30.1%

---

Loyal Customers

2384 customers

Average churn probability:

0.21

Actual churn:

439

Churn rate:

18.4%

---

Champions

1395 customers

Average churn probability:

0.08

Actual churn:

91

Churn rate:

6.5%

---

Business Conclusion:

At Risk customers are the optimal retention target.

High enough churn risk.

Sufficient economic value.

Strong retention ROI.

---

# Visualisations Completed

RFM Segment Churn Rate

Purpose:

Proportion story.

---

RFM Segment Churn Count

Purpose:

Volume story.

---

Dual-Axis Combined Chart

Purpose:

Rate and volume simultaneously.

Used twinx().

Standard business reporting approach.

---

Feature Importance Chart

Generated from:

rf.feature_importances_

Top Features:

1. Satisfaction Score
2. Contract Month-to-Month
3. Tenure in Months
4. Number of Referrals
5. Contract Two Year

Key Insight:

Customer satisfaction is the dominant churn driver.

Churn appears to be primarily an experience problem.

---

Confusion Matrix

True Negatives:

1011

True Positives:

340

False Negatives:

34

False Positives:

24

Recall:

91%

Model misses relatively few churners.

Business cost profile is acceptable.

---

# Files Generated

master_df.csv

rfm_segments.csv

rf_churn_model.pkl

feature_importance.png

confusion_matrix.png

rfm_churn_rate.png

rfm_churn_count.png

rfm_churn_combined.png

---

# Current State

Current notebook:

05_cltv_analysis.ipynb

Notebook exists.

Project has transitioned from churn prediction into customer value analysis.

All prior stages completed.

Random Forest model finalized.

Business action matrix completed.

Visualizations completed.

Feature importance completed.

Confusion matrix completed.

---

# High-Level Remaining Roadmap

The roadmap should be determined by portfolio value and professional standards.

Likely remaining stages include:

1. CLTV Analysis
2. Customer Value Strategy
3. Model Explainability (SHAP or equivalent)
4. Production Pipeline Refactor
5. Dashboard/Application Layer
6. Deployment
7. Repository Cleanup
8. Professional README
9. Portfolio Packaging
10. Interview Preparation Material

Claude should evaluate and adjust this roadmap based on maximum portfolio impact.

---

# Next Step
CLTV isntt useful so we'll move forward with deplyoment with what we have yeah from step 4

