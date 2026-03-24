Student Stress & Performance Predictor  |  PRD v1.0  |  24CAI0203

**PRODUCT REQUIREMENTS DOCUMENT**

*Version 1.0  ·  13 March 2026*

**Student Stress & Academic Performance Predictor**

*Predicting Academic Outcomes and Discovering Lifestyle Archetypes*

*Using Supervised & Unsupervised Machine Learning*

|**Course:**|24CAI0203 — SUL|
| :- | :- |
|**Team:**|Hritabrata Das (Lead) · Harshit · Abhinav Dadwal|
|**Group:**|G-10  |  BE CSE AI&ML  |  4th Sem  |  Batch 2024|
|**Eval 1:**|23–27 March 2026|
|**Final Eval:**|27 April – 1 May 2026|


# **1. Project Overview**
## **1.1  Problem Statement**
Engineering students often experience declining academic performance due to poor lifestyle habits — inadequate sleep, high screen time, low physical activity, and chronic stress. Despite this being widely documented, most institutions have no data-driven early-warning system to identify at-risk students before their grades deteriorate.

This project builds a two-track ML system:

- **Supervised track:** Predict a student's performance category (At Risk vs Performing) from their lifestyle and stress features.
- **Unsupervised track:** Discover natural lifestyle archetypes among students using clustering, without relying on predefined labels.
- **Live predictor:** A web dashboard where any student can enter their habits and receive an instant risk prediction.

## **1.2  Goals & Success Criteria**

|**#**|**Goal**|**Success Metric**|**Priority**|
| :- | :- | :- | :- |
|G1|Collect 50+ primary survey responses|≥50 valid, non-duplicate form submissions|**P0 — Critical**|
|G2|Train 4 supervised classifiers and compare|Best model F1-score ≥ 0.75|**P0 — Critical**|
|G3|K-Means clustering with interpretable clusters|Silhouette score ≥ 0.40, 3 named clusters|**P0 — Critical**|
|G4|Live predictor web dashboard|Working FastAPI + frontend, demo-able at eval|**P1 — High**|
|G5|Full project report submitted|All 6 report sections completed per notice|**P1 — High**|
|G6|Viva-ready team — every member can explain the algorithms|All 3 members can answer the 6 viva questions|**P1 — High**|

## **1.3  Scope**
**In scope**

- Primary data collection via Google Form (15 questions, 50–100 responses)
- Data preprocessing pipeline: null handling, encoding, normalisation, SMOTE
- 4 supervised classifiers: Logistic Regression, Decision Tree, Random Forest, SVM
- Unsupervised: K-Means clustering + PCA for dimensionality reduction and visualisation
- Comparative evaluation with 6 metrics per model
- FastAPI backend serving the trained model as a REST API
- React/HTML frontend dashboard with 4 tabs: Dataset Overview, Model Comparison, Cluster Explorer, Live Predictor
- Full project report document

**Out of scope**

- Deep learning / neural networks (not required by 24CAI0203 syllabus)
- User accounts, authentication, or persistent storage of predictions
- Mobile app (web responsive is sufficient for demo)
- Deployment to production cloud (Oracle VM hosting is optional bonus)


# **2. Dataset Specification**
## **2.1  Data Collection**

|**Attribute**|**Details**|
| :- | :- |
|**Method**|Google Form — anonymous, no PII collected|
|**Target sample**|50–100 BE CSE AI&ML students, Batches G-06 to G-12, Chitkara University|
|**Distribution**|Share via WhatsApp batch groups. Target: responses within 5 days of form going live.|
|**Export format**|Google Sheets → Download as CSV → load with pandas|
|**Imbalance handling**|If At Risk class < 30% of total, apply SMOTE on training split only (never on test split)|
|**Privacy**|Form collects zero PII. No names, roll numbers, emails, or device identifiers.|

## **2.2  Feature Schema (15 features)**

|**#**|**Feature**|**Column name**|**Type**|**Range / Values**|
| :- | :- | :- | :- | :- |
|1|Current CGPA|cgpa|Float|4\.0 – 10.0|
|2|Attendance %|attendance\_pct|Float|0 – 100|
|3|Daily study hours|study\_hours|Ordinal|<2 / 2–4 / 4–6 / >6  →  encode: 1/2/3/4|
|4|Sleep hours/night|sleep\_hrs|Integer|4 – 12|
|5|Screen time (hrs)|screen\_time|Float|0 – 18|
|6|Physical activity days/week|activity\_days|Integer|0 – 7|
|7|Extracurricular activities|extracurricular|Ordinal|None/1/2/3+  →  encode: 0/1/2/3|
|8|Part-time work/freelancing|part\_time\_work|Binary|Yes/No  →  1/0|
|9|Accommodation type|accommodation|Binary|Hostel/Day Scholar  →  1/0|
|10|Phone after midnight|phone\_midnight|Ordinal|Rarely/Sometimes/Often  →  0/1/2|
|11|Academic stress level|stress\_level|Integer|1 – 10|
|12|Overwhelmed by workload|overwhelmed|Ordinal|Never/Sometimes/Often/Always  →  0/1/2/3|
|13|Social support rating|social\_support|Integer|1 – 5|
|14|Skips meals due to workload|skips\_meals|Ordinal|Never/Rarely/Often  →  0/1/2|
|15|Life satisfaction rating|life\_satisfaction|Integer|1 – 10|

## **2.3  Target Variable Engineering**
The CGPA column is used ONLY to create the target label. It is then dropped before model training.

|<p>**Python: Label creation**</p><p>df['label'] = df['cgpa'].apply(lambda x: 1 if x >= 7.0 else 0)</p><p># 1 = Performing  |  0 = At Risk</p><p>df.drop(columns=['cgpa'], inplace=True)  # remove raw CGPA from features</p><p></p><p># Optional 3-class variant:</p><p># Low: cgpa < 6.5  |  Average: 6.5–7.9  |  High: ≥ 8.0</p>|
| :- |


# **3. Data Preprocessing Pipeline**
## **3.1  Step-by-Step Pipeline**

|**Step**|**Stage**|**Action**|**Code / Tool**|
| :- | :- | :- | :- |
|1|Load data|Read CSV exported from Google Sheets|pd.read\_csv('survey.csv')|
|2|Rename columns|Map raw form headers to clean snake\_case names|df.rename(columns={...})|
|3|Null audit|Drop rows with >30% missing; median/mode impute rest|df.isnull().sum() → df.fillna()|
|4|Outlier removal|IQR filter on numerical columns (cgpa, sleep, screen\_time)|IQR = Q3 - Q1; drop > Q3+1.5\*IQR|
|5|Label creation|Create binary label from CGPA, then drop CGPA|See Section 2.3|
|6|Ordinal encoding|Map categorical features to integers (study\_hours, overwhelmed, etc.)|Manual map dict + df.replace()|
|7|Train-test split|80% train / 20% test, stratified on label|train\_test\_split(stratify=y)|
|8|SMOTE|Oversample minority class on TRAIN split only|from imblearn.over\_sampling import SMOTE|
|9|Scaling|StandardScaler fit on train, transform both train and test|StandardScaler().fit\_transform(X\_train)|
|10|EDA outputs|Correlation heatmap, class distribution bar, pairplot|seaborn.heatmap(), sns.pairplot()|

|<p>**Critical rule: SMOTE and StandardScaler must be fit ONLY on X\_train, then applied to X\_test.**</p><p>Fitting on the full dataset causes data leakage and artificially inflates metrics.</p>|
| :- |


# **4. Machine Learning Architecture**
## **4.1  Supervised Learning Track**
Train and compare 4 classifiers on identical train/test splits. GridSearchCV with 5-fold cross-validation for hyperparameter tuning.

|**Algorithm**|**Library**|**Key Hyperparameters to Tune**|**Expected Role**|
| :- | :- | :- | :- |
|Logistic Regression|sklearn.linear\_model|C: [0.01, 0.1, 1, 10], penalty: ['l1','l2'], solver: 'liblinear'|Baseline. Simplest, most interpretable.|
|Decision Tree|sklearn.tree|max\_depth: [3,5,7,10], min\_samples\_split: [2,5,10], criterion: ['gini','entropy']|Interpretable rules, prone to overfit.|
|Random Forest|sklearn.ensemble|n\_estimators: [50,100,200], max\_depth: [5,10,None], max\_features: ['sqrt','log2']|Main model. Best expected accuracy.|
|SVM|sklearn.svm|C: [0.1,1,10], kernel: ['linear','rbf'], gamma: ['scale','auto']|Strong on small datasets. Good margin.|

**Evaluation metrics (report ALL of these for each model)**

|**Metric**|**Formula**|**Why It Matters Here**|
| :- | :- | :- |
|Accuracy|(TP+TN) / Total|Overall correctness. Report but don't rely on alone if class imbalance exists.|
|Precision|TP / (TP+FP)|How many predicted At Risk were actually At Risk.|
|Recall|TP / (TP+FN)|How many actual At Risk students did we catch. More important than precision here.|
|F1-Score|2 × (P×R)/(P+R)|Primary comparison metric. Balances precision and recall.|
|ROC-AUC|Area under ROC curve|Model discrimination ability across thresholds.|
|Confusion Matrix|[[TN,FP],[FN,TP]]|Visual heatmap for each model. Required for viva.|

## **4.2  Unsupervised Learning Track**
**K-Means Clustering**

- Input: all 14 features (after preprocessing, excluding label). Apply PCA first to reduce to 2–3 components.
- Use Elbow Method to find optimal k: plot inertia vs k=2 to k=8. Pick the 'elbow' point (typically k=3).
- Validate with Silhouette Score. Target: ≥ 0.40.
- After clustering, compute mean feature values per cluster → name each cluster based on dominant pattern.

**Expected cluster archetypes**

|**Cluster Name**|**Dominant pattern**|**Sleep**|**Stress**|**Likely label overlap**|
| :- | :- | :- | :- | :- |
|Balanced Achiever|Good sleep, low stress, regular activity|7–8 hrs|3–5/10|Mostly Performing|
|Sleep-Deprived Grinder|High study hrs, low sleep, high stress|4–5 hrs|6–8/10|Mixed — some At Risk|
|Overloaded & Burned Out|High stress, low activity, skips meals|<5 hrs|8–10/10|Mostly At Risk|

**PCA specifics**

- Fit PCA on scaled X\_train. Choose n\_components to retain ≥ 80% cumulative explained variance.
- Plot scree plot (explained variance ratio per component).
- Use 2D PCA projection to visualise cluster separation — scatter plot coloured by cluster label.


# **5. Technical Stack**
## **5.1  Full Stack Overview**

|**Layer**|**Technology**|**Version**|**Purpose**|
| :- | :- | :- | :- |
|Data|Python|3\.11+|All ML, preprocessing, analysis|
||pandas|2\.x|Data loading, cleaning, manipulation|
||NumPy|1\.26+|Numerical operations|
|ML|scikit-learn|1\.4+|All classifiers, clustering, metrics, pipelines|
||imbalanced-learn|0\.12+|SMOTE for class imbalance|
||joblib|built-in|Model serialisation (pickle replacement)|
|Visualisation|Matplotlib|3\.8+|All plots and charts|
||Seaborn|0\.13+|Heatmaps, pairplots, styled charts|
|Backend|FastAPI|0\.111+|REST API serving the trained model|
||Uvicorn|0\.29+|ASGI server for FastAPI|
||Pydantic v2|2\.x|Request/response schema validation|
|Frontend|React (Vite) OR plain HTML+JS|18\.x / vanilla|Dashboard UI — 4 tabs|
||Chart.js|4\.x|Bar charts, radar charts for model comparison|
|Notebook|Jupyter Notebook|7\.x|All ML experiments, EDA, model training|
|Environment|conda / venv|—|Isolated Python environment|

## **5.2  Python Environment Setup**

|<p>**# Create and activate virtual environment**</p><p>python -m venv venv</p><p>source venv/bin/activate  # Windows: venv\Scripts\activate</p><p></p><p># Install all dependencies</p><p>pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn</p><p>pip install fastapi uvicorn[standard] pydantic joblib</p><p>pip install jupyter notebook</p><p></p><p># Freeze for reproducibility</p><p>pip freeze > requirements.txt</p>|
| :- |


# **6. Project File Structure**

|<p>**student\_stress\_predictor/**</p><p>├── data/</p><p>│   ├── raw/                  # Original CSV from Google Form</p><p>│   │   └── survey\_raw.csv</p><p>│   └── processed/            # After preprocessing</p><p>│       ├── X\_train.csv</p><p>│       ├── X\_test.csv</p><p>│       ├── y\_train.csv</p><p>│       └── y\_test.csv</p><p>├── notebooks/</p><p>│   ├── 01\_EDA.ipynb          # Exploratory data analysis</p><p>│   ├── 02\_preprocessing.ipynb</p><p>│   ├── 03\_supervised.ipynb   # All 4 classifiers</p><p>│   ├── 04\_unsupervised.ipynb # K-Means + PCA</p><p>│   └── 05\_comparison.ipynb   # Side-by-side results</p><p>├── models/</p><p>│   ├── logistic\_regression.pkl</p><p>│   ├── decision\_tree.pkl</p><p>│   ├── random\_forest.pkl     # Primary model for API</p><p>│   ├── svm.pkl</p><p>│   ├── kmeans.pkl</p><p>│   ├── pca.pkl</p><p>│   └── scaler.pkl            # StandardScaler — must save this</p><p>├── api/</p><p>│   ├── main.py               # FastAPI app</p><p>│   ├── schemas.py            # Pydantic request/response models</p><p>│   └── predictor.py          # Model loading + inference logic</p><p>├── frontend/</p><p>│   ├── index.html            # Dashboard HTML</p><p>│   ├── app.js                # Tab logic + Chart.js</p><p>│   └── style.css</p><p>├── report/                   # Final project report (Word doc)</p><p>├── requirements.txt</p><p>└── README.md</p>|
| :- |


# **7. API Design (FastAPI)**
## **7.1  Endpoints**

|**Method**|**Endpoint**|**Purpose**|**Request / Response**|
| :- | :- | :- | :- |
|**GET**|/health|Health check|Response: {"status": "ok", "model": "random\_forest"}|
|**POST**|/predict|Predict performance label for new student input|Body: StudentInput schema → Response: PredictionResult schema|
|**GET**|/metrics|Return stored model evaluation metrics|Response: {model\_name: {accuracy, f1, precision, recall, roc\_auc}}|
|**GET**|/clusters|Return cluster centroids and names|Response: {clusters: [{id, name, centroid\_values}]}|
|**GET**|/feature-importance|Return RF feature importance scores|Response: [{feature, importance}] sorted desc|

## **7.2  Pydantic Schemas**

|<p>**# schemas.py**</p><p>from pydantic import BaseModel, Field</p><p></p><p>class StudentInput(BaseModel):</p><p>`    `attendance\_pct:   float = Field(..., ge=0, le=100)</p><p>`    `study\_hours:      int   = Field(..., ge=1, le=4)      # 1=<2hrs, 2=2-4, 3=4-6, 4=>6</p><p>`    `sleep\_hrs:        float = Field(..., ge=4, le=12)</p><p>`    `screen\_time:      float = Field(..., ge=0, le=18)</p><p>`    `activity\_days:    int   = Field(..., ge=0, le=7)</p><p>`    `extracurricular:  int   = Field(..., ge=0, le=3)</p><p>`    `part\_time\_work:   int   = Field(..., ge=0, le=1)</p><p>`    `accommodation:    int   = Field(..., ge=0, le=1)</p><p>`    `phone\_midnight:   int   = Field(..., ge=0, le=2)</p><p>`    `stress\_level:     int   = Field(..., ge=1, le=10)</p><p>`    `overwhelmed:      int   = Field(..., ge=0, le=3)</p><p>`    `social\_support:   int   = Field(..., ge=1, le=5)</p><p>`    `skips\_meals:      int   = Field(..., ge=0, le=2)</p><p>`    `life\_satisfaction:int   = Field(..., ge=1, le=10)</p><p></p><p>class PredictionResult(BaseModel):</p><p>`    `label:       int    # 0 = At Risk, 1 = Performing</p><p>`    `label\_text:  str    # 'At Risk' or 'Performing'</p><p>`    `confidence:  float  # predict\_proba score of predicted class</p><p>`    `cluster\_id:  int    # K-Means cluster assignment</p><p>`    `cluster\_name:str    # e.g. 'Balanced Achiever'</p>|
| :- |

## **7.3  Predictor Logic**

|<p>**# predictor.py**</p><p>import joblib, numpy as np</p><p>from schemas import StudentInput</p><p></p><p>scaler = joblib.load('models/scaler.pkl')</p><p>model  = joblib.load('models/random\_forest.pkl')</p><p>kmeans = joblib.load('models/kmeans.pkl')</p><p>pca    = joblib.load('models/pca.pkl')</p><p></p><p>CLUSTER\_NAMES = {0: 'Balanced Achiever', 1: 'Sleep-Deprived Grinder', 2: 'Overloaded & Burned Out'}</p><p>FEATURE\_ORDER = ['attendance\_pct','study\_hours','sleep\_hrs','screen\_time',</p><p>`                 `'activity\_days','extracurricular','part\_time\_work','accommodation',</p><p>`                 `'phone\_midnight','stress\_level','overwhelmed','social\_support',</p><p>`                 `'skips\_meals','life\_satisfaction']</p><p></p><p>def predict(inp: StudentInput):</p><p>`    `x = np.array([[getattr(inp, f) for f in FEATURE\_ORDER]])</p><p>`    `x\_scaled = scaler.transform(x)</p><p>`    `label = int(model.predict(x\_scaled)[0])</p><p>`    `conf  = float(model.predict\_proba(x\_scaled)[0][label])</p><p>`    `x\_pca = pca.transform(x\_scaled)</p><p>`    `cluster = int(kmeans.predict(x\_pca)[0])</p><p>`    `return {'label': label, 'label\_text': 'Performing' if label==1 else 'At Risk',</p><p>`            `'confidence': round(conf,3), 'cluster\_id': cluster,</p><p>`            `'cluster\_name': CLUSTER\_NAMES[cluster]}</p>|
| :- |


# **8. Frontend Dashboard**
## **8.1  Tab Specifications**

|**Tab**|**Content**|**Data source**|**Chart type**|
| :- | :- | :- | :- |
|Dataset Overview|Class distribution, feature distributions, correlation heatmap, response count|Static JSON embedded at build time from EDA notebook|Bar chart (class dist.), heatmap table, pie chart|
|Model Comparison|Side-by-side F1, Accuracy, Precision, Recall, ROC-AUC for all 4 models|GET /metrics API endpoint|Grouped bar chart (Chart.js), metrics table|
|Cluster Explorer|3 cluster cards: name, centroid values, dominant features, % of dataset|GET /clusters API endpoint|Radar chart per cluster, 2D PCA scatter (static image from notebook)|
|Live Predictor|14-field input form → POST /predict → show label, confidence bar, cluster badge|POST /predict API endpoint|Confidence bar, result card, cluster pill badge|

## **8.2  Live Predictor UX Flow**
1. User enters 14 lifestyle values into the input form (sliders + dropdowns).
1. On submit, frontend validates all fields client-side (range checks).
1. POST request sent to /predict with JSON body (StudentInput schema).
1. Response received: label, confidence, cluster\_id, cluster\_name.
1. Dashboard renders result card: 'At Risk' (red) or 'Performing' (green) with confidence percentage.
1. Cluster badge displayed below: cluster name + short description of what that cluster means.

|<p>**Demo tip: Prepare 3 preset input profiles for the viva day —**</p><p>Profile A: 8hrs sleep, stress 3, attendance 85% → should predict Performing</p><p>Profile B: 4hrs sleep, stress 9, attendance 55% → should predict At Risk</p><p>Profile C: 6hrs sleep, stress 6, attendance 70% → edge case, shows confidence score</p>|
| :- |


# **9. Sprint Plan**

|**Sprint**|**Dates**|**Tasks**|**Owner**|
| :- | :- | :- | :- |
|Sprint 0 Setup|Mar 13–14|<p>Send Google Form to batch</p><p>Get lab teacher approval on problem statement</p><p>Set up Python venv + install all dependencies</p><p>Create GitHub repo + folder structure</p>|Hritabrata|
|Sprint 1 Data|Mar 15–20|<p>Collect ≥50 form responses</p><p>Export CSV, rename columns to schema names</p><p>Run null audit + outlier removal</p><p>Create target label, encode categoricals</p>|All members|
|Sprint 2 Eval 1 Prep|Mar 21–22|<p>Complete 01\_EDA.ipynb: heatmap, distributions, class balance</p><p>Prepare 3-slide eval deck: problem, dataset, ML approach</p><p>Do a dry run — all 3 members practise verbal explanation</p>|Hritabrata leads|
|EVAL 1|Mar 23–27|<p>Present: problem understanding, dataset attributes, feasibility, ML approach</p><p>Target: full 2/2 marks</p>|All members|
|Sprint 3 Supervised ML|Mar 28–Apr 7|<p>02\_preprocessing.ipynb: full pipeline with SMOTE</p><p>03\_supervised.ipynb: train all 4 classifiers with GridSearchCV</p><p>Generate confusion matrices, ROC curves, feature importance plot</p><p>Fill in metrics comparison table</p>|Hritabrata + Abhinav|
|Sprint 4 Unsupervised|Apr 8–13|<p>04\_unsupervised.ipynb: PCA + elbow method + K-Means</p><p>Generate 2D cluster scatter plot</p><p>Name clusters, compute cluster stats</p><p>05\_comparison.ipynb: full comparison summary</p>|Harshit + Hritabrata|
|Sprint 5 API + Dashboard|Apr 14–20|<p>Build FastAPI: all 5 endpoints + schemas + predictor.py</p><p>Save model artifacts (scaler.pkl, random\_forest.pkl, kmeans.pkl, pca.pkl)</p><p>Build frontend: 4 tabs, Chart.js visualisations, live predictor form</p><p>End-to-end test: form submit → API → result rendered</p>|Hritabrata|
|Sprint 6 Report + Viva|Apr 21–26|<p>Write full project report (all 6 sections per notice)</p><p>Prepare architecture diagram (draw.io or Python matplotlib)</p><p>Full viva mock: each member answers 6 questions cold</p><p>Final dashboard polish + demo script</p>|All members|
|FINAL EVAL|Apr 27–May 1|<p>Live dashboard demo</p><p>Project report submitted</p><p>Viva — all 3 members answer independently</p><p>Target: full 8/8 remaining marks</p>|All members|


# **10. Team Responsibilities**

|**Member**|**Primary Role**|**Key Deliverables**|
| :- | :- | :- |
|**Hritabrata Das (Group Leader)**|ML Lead + Backend|<p>Sprint planning and GitHub repo management</p><p>Supervised ML notebook (03\_supervised.ipynb)</p><p>FastAPI backend (all endpoints)</p><p>Final dashboard integration</p><p>Project report compilation</p>|
|**Harshit**|Data + Unsupervised|<p>Google Form distribution and response collection</p><p>Preprocessing pipeline (02\_preprocessing.ipynb)</p><p>K-Means + PCA notebook (04\_unsupervised.ipynb)</p><p>Cluster interpretation and naming</p><p>EDA visualisations</p>|
|**Abhinav Dadwal**|Analysis + Report|<p>EDA notebook (01\_EDA.ipynb)</p><p>Comparison notebook (05\_comparison.ipynb)</p><p>Metrics table and results write-up</p><p>Architecture diagram</p><p>Viva preparation slides</p>|

|<p>**All 3 members must be able to explain every algorithm and every result independently.**</p><p>Viva is 5/10 marks. You lose marks if one member cannot answer — the examiner will ask individuals.</p>|
| :- |

# **11. Risks & Mitigations**

|**#**|**Risk**|**Likelihood**|**Impact**|**Mitigation**|
| :- | :- | :- | :- | :- |
|R1|Low survey response count (<40)|Medium|High|Send reminders, use SMOTE, approach neighbouring batches|
|R2|Severe class imbalance in CGPA labels|Medium|High|SMOTE on train split; report F1 not accuracy|
|R3|Low model accuracy due to small dataset|Medium|Medium|Expected — document it as a limitation. Focus on comparison, not raw accuracy.|
|R4|FastAPI / frontend not ready for demo|Low|Medium|Fallback: run prediction live in Jupyter Notebook during demo|
|R5|Team member unprepared for viva|Low|High|Weekly 20-min verbal mock sessions starting Apr 14|

# **12. Project Report Outline**
Per the project notice (Ref CUIET/CSE/i-AI/2026/016), the report must include all 6 sections below. Use this as your writing checklist.

|**#**|**Section**|**What to Include**|
| :- | :- | :- |
|a|Problem Definition|Real-world motivation, problem statement, objectives, ML approach summary, why this dataset|
|b|System Architecture / Flowchart|Architecture diagram showing full pipeline from survey → preprocessing → supervised/unsupervised tracks → dashboard|
|c|Methodology Description|Step-by-step preprocessing pipeline, justification for each algorithm choice, hyperparameter tuning approach|
|d|Dataset Description & Preparation|All 15 features with types and ranges, collection method, null handling, encoding decisions, SMOTE justification|
|e|Comparative Results|Metrics table for all 4 supervised models, confusion matrix plots, feature importance, elbow curve, silhouette score, cluster scatter|
|f|Limitations of the Model|Self-report bias, small sample size, no causation (only correlation), static snapshot, single institution|

|<p>**You're ready to start. First action: send the Google Form link to your batch today.**</p><p>Everything else in this PRD follows once you have data.</p>|
| :- |

Page   |  Hritabrata Das · Harshit · Abhinav Dadwal  |  Group G-10
