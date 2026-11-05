# Fake News Detection XAI

A Streamlit application that analyzes a news claim or article using a text-classification model, web search, evidence ranking, natural-language inference (NLI), and an interpretable ML explanation.

The app presents a final **REAL**, **FAKE**, or **UNVERIFIED** verdict. The ML prediction is an initial signal; the final verdict is based on retrieved evidence and NLI verification.

## Features

- Classifies article text as **REAL** or **FAKE** with a calibrated Linear SVM.
- Extracts simple claim metadata: named entities, dates, and claim type.
- Generates search queries and retrieves results through the Tavily Search API.
- Filters and ranks evidence using claim overlap, source quality, and date relevance.
- Checks relevant evidence sentences with the `facebook/bart-large-mnli` model.
- Aggregates supporting and refuting evidence to produce a final verdict.
- Uses LIME to show which words influenced the ML classification.
- Displays evidence sources and model comparison metrics in the Streamlit interface.

## How the pipeline works

1. The input text is classified by the trained TF-IDF and calibrated Linear SVM model.
2. The claim analyzer extracts entities, dates, and a broad claim type, which are used to generate search queries.
3. Tavily search results are deduplicated, filtered by keyword overlap, and ranked.
4. Up to five top-ranked results are checked by the NLI model for support, contradiction, or neutrality.
5. Evidence that meets the score and confidence thresholds is aggregated. Stronger consistent support produces **REAL**, stronger consistent refutation produces **FAKE**, and insufficient evidence produces **UNVERIFIED**.
6. LIME explains the separate ML prediction; its explanation does not determine the final evidence-based verdict.

## Project structure

```text
.
├── app.py                         # Streamlit application
├── config/
│   └── settings.py                # Project paths and shared thresholds
├── data/
│   ├── raw/                       # Fake.csv and True.csv input datasets
│   └── processed/                 # Prepared news_dataset.csv
├── models/                        # Trained model and TF-IDF vectorizer
├── src/
│   ├── decision/                  # Final verdict logic
│   ├── evidence/                  # Filtering, ranking, NLI, aggregation
│   ├── explainability/            # LIME explanations
│   ├── ml/                        # Dataset preparation, training, prediction
│   ├── nlp/                       # Claim analysis and query generation
│   └── retrieval/                 # Web search and retrieval
├── tests/                         # Pytest test suite
└── requirements.txt
```

## Requirements

- Python with `pip`
- A Tavily API key for web evidence retrieval
- Internet access for Tavily searches and the first download of the NLI model

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Tavily API key:

```text
TAVILY_API_KEY=your_tavily_api_key
```

The key is read by the web-search integration. Without it, search results cannot be retrieved and the app may return **UNVERIFIED** when it has no evidence.

## Data and model files

The training scripts expect these input files:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Each CSV is expected to contain `title` and `text` columns. Prepare the combined dataset and train the deployed model:

```powershell
python -m src.ml.prepare_dataset
python -m src.ml.train
```

Dataset preparation writes `data/processed/news_dataset.csv`. Training writes the following files, which the app loads for ML predictions:

```text
models/fake_news_model.joblib
models/tfidf_vectorizer.joblib
```

The trainer uses a TF-IDF vectorizer with unigrams and bigrams and a calibrated Linear SVM. Raw/processed datasets and serialized model files are excluded by `.gitignore`; provide or generate them separately when setting up a fresh checkout.

To compare Logistic Regression, Linear SVM, and Multinomial Naive Bayes on the prepared dataset, run:

```powershell
python -m src.ml.compare_models
```

## Run the application

With dependencies and model files available:

```powershell
streamlit run app.py
```

Enter a claim or article in the text area and select **Verify News**. The app displays the final verdict, ML prediction and confidence, claim analysis, search queries, evidence details, and LIME feature weights.

## Tests

Run the test suite from the project root:

```powershell
pytest
```

## Limitations

- This is an experimental decision-support tool, not an authoritative fact-checking service. Verify important claims using reliable primary sources.
- ML confidence is not a guarantee of factual accuracy. The training data may contain dataset-specific patterns or biases.
- Claim entity/date extraction and evidence relevance scoring use lightweight heuristics.
- Web search coverage depends on Tavily and the available indexed sources. Missing, weak, or conflicting evidence can result in **UNVERIFIED**.
- LIME explains the ML text classifier only; it does not explain the evidence-based final verdict.
- The model comparison figures shown in the app are currently defined in `app.py` as fixed values and are not recalculated at runtime.
