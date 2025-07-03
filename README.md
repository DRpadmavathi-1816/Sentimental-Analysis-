o# Sentiment Analysis with TF-IDF & Logistic Regression

A lightweight end-to-end notebook that classifies customer‐generated text (tweets/reviews) as **positive** or **negative**.  
The workflow covers data loading, preprocessing, TF-IDF vectorization, model training, and performance evaluation.

INTERNSHIP INFORMATION 
- Name: D.Renuka Padmavathi 
- company: Micro IT SOLUTIONS 
- Domain : ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING 
- Duration : 1 Month 

## 📂 Project Contents

|File / Folder                               Purpose                                                    
 `sentiment_anaook with full code (imports → plots). 
`README.md`                                   You’re reading it—setup & usage guide.                     
`requirements.txt`                    Reproducible Python dependencies (optional).            


## 📊 Dataset

The sample notebook fetches the **Twitter Sentiment Analysis** training set  
`https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv`,  
containing two columns:

| Column   | Description                                
| `tweet`  | Raw tweet text                             |
| `label`  | Sentiment (0 = negative, 1 = positive)     |

**Using your own data?** Replace the `url` with a local/remote CSV that has  
`review` (text) and `sentiment` (label) columns—or adapt the renaming sec

## 🖥️ Quick Start

1. **Clone / download** the project or copy the notebook.
2. 🔧 **Install dependencies**

   ```bash
   # Create (optional) virtual environment
   python -m venv .venv && source .venv/bin/activate   # macOS / Linux
   # .venv\Scripts\Activate.ps1                        # Windows PowerShell

   # Core libraries
   pip install -r requirements.txt
   # OR, minimal install
   pip install pandas numpy scikit-learn seaborn matplotlib nltk

3. Launch Jupyter

jupyter notebook sentiment_analysis.ipynb


4. Run all cells (Kernel ▸ Restart & Run All) – the notebook download

🔍 Workflow Breakdown

1. Imports & Config – pandas, NumPy, Matplotlib, Seaborn, scikit-learn, NLTK.


2. Data Loading – read CSV → rename columns → map labels to text.


3. Preprocessing

Lower-casing, URL / mention / hashtag removal

Punctuation stripping

Stop-word filtering

Porter stemming



4. Feature Extraction – scikit-learn TfidfVectorizer (top 5 000 tokens).


5. Model Training – LogisticRegression() on 80 / 20 train-test split.


6. Evaluation

Accuracy score

Classification report (precision, recall, F1)

Confusion matrix heat-map (Seaborn)



7. Next Steps (optional)

- Hyper-parameter tuning (GridSearchCV)

- Cross-validation (StratifiedKFold)

- Save & deploy model (joblib, Flask/FastAPI)

- Real-time inference pipeline or Streamlit app.



🛠️ Customising for New Data

1. Ensure your CSV has a text column and binary label (0/1 or negative/positive).


2. Update the url (or pd.read_csv("path/to/your.csv")).


3. Tweak preprocess_text() – e.g., add lemmatisation, emoji handling, language filters.


4. Adjust max_features in TfidfVectorizer or switch to character-level n-grams.


5. Experiment with other algorithms: SVM, Naïve Bayes, XGBoost, BERT embeddings.




---

⚠️ Troubleshooting

Issue	Fix

LookupError: stopwords not found	            The notebook calls nltk.download('stopwords')—ensure the cell runs once with internet
ConvergenceWarning in Logistic Regression	Increase max_iter (e.g., LogisticRegression(max_iter=1000))
ModuleNotFoundError: seaborn	pip install seaborn


Twitter Sentiment Analysis dataset – Kaggle / GitHub community.

scikit-learn, NLTK, pandas, Matplotlib, Seaborn – open-source libraries.


