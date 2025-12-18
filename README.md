# RealFakeNewsDetector

- To create a virtual environment, run "python3 -m venv venv", then activate it by running "source/venv/bins/activate"
    - Then run: "pip install -r requirements.txt" to download all dependencies
- Raw datasets are in datasets/, datasets/processed contains combined datasets 
- pipeline.py contains all core processing + model utilities:
    1. Data Loading
        -  Data loading: `load_kaggle()`, `load_politifact()`, `load_gossipcop()` (will combine CSVs when needed)
        - Dataset helpers: `combine_dataset()` to merge fake/real CSVs into one file, 
        - `datasets/` — Raw dataset directories (Kaggle, PolitiFact, GossipCop). Use `pipeline.py` helpers or the notebooks to combine/prepare these.
    2. Text Cleaning
	    - clean_text() lowercases, strips punctuation, removes URLs, expands contractions
	    - clean_dataset() cleans + deduplicates titles and article bodies
    3. Training + Evaluation
	    - Baseline training: `train_logreg()`, `train_svm()`, `train_nb()` and `eval_on_dataset()` for evaluating models
	    - eval_on_dataset() evaluates on any dataset (Kaggle, PolitiFact, GossipCop)
- Notebooks (open in Jupyter or run through IDE)
    - `LSTMmodel.ipynb` — LSTM-based model training and evaluation; includes preprocessing and how to save/load `lstm_model.pt`.
    - `distilbertModel.ipynb` , `LSTMmodel.ipynb` — DistilBERT and LSTM experiments and results
    - `gossipCopModel.ipynb`, `kaggleModel.ipynb`, `politifactModel.ipynb` — Dataset-specific experiments and evaluation harnesses for each dataset.
- All trained baseline models and vectorizers are saved inside the joblist/ folder (trained on the Kaggle dataset)