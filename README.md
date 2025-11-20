# RealFakeNewsDetector

- To activate virtual environment: run "source/venv/bins/activate"
    - Then run: "pip install -r requirements.txt" to download all dependencies
- Raw datasets are in datasets/, datasets/processed contains cleaned datasets 
- pipeline.py contains all core processing + model utilities:
    1. Data Loading
        - load_kaggle() downloads and loads the Kaggle Fake News dataset
        - combine_politifact() merges PolitiFact fake/real CSVs into one processed file
    2. Text Cleaning
	    - clean_text() lowercases, strips punctuation, removes URLs, expands contractions
	    - clean_dataset() cleans + deduplicates titles and article bodies
    3. Training + Evaluation
	    - train_logreg() trains a TF-IDF + Logistic Regression baseline
	    - eval_on_dataset() evaluates on any dataset (Kaggle, PolitiFact, GossipCop)
- All trained models and vectorizers are saved inside the joblist/ folder (trained on the Kaggle dataset)