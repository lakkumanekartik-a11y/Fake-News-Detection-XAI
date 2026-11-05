import streamlit as st
import pandas as pd

from src.ml.predict import predict_news
from src.nlp.claim_analyzer import analyze_claim
from src.retrieval.retriever import retrieve_evidence

from src.evidence.evidence_filter import filter_evidence
from src.evidence.evidence_ranker import rank_evidence
from src.evidence.nli_verifier import verify_claim
from src.evidence.evidence_aggregator import aggregate_evidence

from src.decision.decision_engine import decide_verdict

from src.explainability.xai_explainer import explain_prediction


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Fake News Detection XAI",
    page_icon="🔎",
    layout="wide",
)


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🔎 Fake News Detection using ML, NLP & XAI"
)

st.write(
    "Enter a news claim or article text to analyze "
    "its credibility using machine learning, web evidence "
    "and Natural Language Inference."
)


# ==========================================================
# INPUT
# ==========================================================

text = st.text_area(
    "📰 Enter News Claim / Article",
    height=180,
    placeholder=(
        "Example: India launched a new AI policy in 2026."
    ),
)


verify_button = st.button(
    "🔍 Verify News",
    type="primary",
    use_container_width=True,
)


# ==========================================================
# PIPELINE
# ==========================================================

if verify_button:

    # ------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------

    if not text.strip():

        st.warning(
            "Please enter a news claim or article."
        )

        st.stop()


    # ======================================================
    # ML
    # ======================================================

    with st.spinner(
        "Running ML model..."
    ):

        ml_label, ml_confidence = predict_news(
            text
        )


    # ======================================================
    # NLP
    # ======================================================

    with st.spinner(
        "Analyzing claim..."
    ):

        claim_info = analyze_claim(
            text
        )


    # ======================================================
    # RETRIEVAL
    # ======================================================

    with st.spinner(
        "Searching real-world evidence..."
    ):

        retrieval = retrieve_evidence(
            text
        )


    raw_results = retrieval.get(
        "results",
        []
    )


    # ======================================================
    # EVIDENCE FILTER
    # ======================================================

    filtered_results = filter_evidence(
        text,
        raw_results
    )


    # ======================================================
    # EVIDENCE RANKING
    # ======================================================

    ranked_results = rank_evidence(
        text,
        filtered_results
    )


    # Only strongest evidence
    top_results = ranked_results[:5]


    # ======================================================
    # NLI VERIFICATION
    # ======================================================

    verified_results = []

    with st.spinner(
        "Verifying evidence with NLI..."
    ):

        for evidence in top_results:

            evidence_text = " ".join(
                [
                    evidence.get(
                        "title",
                        ""
                    ),
                    evidence.get(
                        "content",
                        ""
                    ),
                ]
            )

            if not evidence_text.strip():
                continue

            try:

                nli_result = verify_claim(
                    text,
                    evidence_text
                )

                item = evidence.copy()

                item.update(
                    nli_result
                )

                verified_results.append(
                    item
                )

            except Exception:
                continue


    # ======================================================
    # EVIDENCE AGGREGATION
    # ======================================================

    aggregated = aggregate_evidence(
        verified_results
    )


    # ======================================================
    # FINAL DECISION
    # ======================================================

    decision = decide_verdict(
        aggregated
    )


    # ======================================================
    # XAI
    # ======================================================

    with st.spinner(
        "Generating ML explanation..."
    ):

        xai_result = explain_prediction(
            text,
            num_features=8
        )


    # ======================================================
    # FINAL VERIFICATION
    # ======================================================

    st.divider()

    st.subheader(
        "🎯 Final Verification"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Final Verdict",
            decision["verdict"]
        )


    with col2:

        st.metric(
            "Confidence",
            f"{decision['confidence']:.2%}"
        )


    st.info(
        decision["reason"]
    )


    # ======================================================
    # MACHINE LEARNING
    # ======================================================

    st.divider()

    st.subheader(
        "🤖 Machine Learning"
    )


    # ------------------------------------------------------
    # SELECTED MODEL
    # ------------------------------------------------------

    st.success(
        "Selected Model: Calibrated Linear SVM"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "ML Prediction",
            ml_label
        )


    with col2:

        st.metric(
            "ML Confidence",
            f"{ml_confidence:.2%}"
        )


    st.caption(
        "The ML model provides an initial classification signal. "
        "The final verdict is determined using external evidence "
        "and NLI verification."
    )


    # ======================================================
    # MODEL COMPARISON
    # ======================================================

    st.subheader(
        "📊 Model Comparison"
    )


    # Results from the same test split
    model_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Linear SVM",
                "Multinomial Naive Bayes",
            ],
            "Accuracy": [
                98.68,
                99.48,
                95.45,
            ],
            "Precision": [
                98.36,
                99.37,
                95.56,
            ],
            "Recall": [
                99.22,
                99.67,
                96.06,
            ],
            "F1 Score": [
                98.79,
                99.52,
                95.81,
            ],
        }
    )


    # ------------------------------------------------------
    # TABLE
    # ------------------------------------------------------

    st.dataframe(
        model_data.style.format(
            {
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1 Score": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ------------------------------------------------------
    # ATTRACTIVE MODEL GRAPH
    # ------------------------------------------------------

    st.write(
        "**Model Performance Comparison**"
    )


    chart_data = model_data.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ]
    ]


    st.bar_chart(
        chart_data,
        use_container_width=True,
        height=400,
    )


    st.success(
        "Linear SVM achieved the highest performance "
        "across the evaluated metrics on the test dataset."
    )


    st.caption(
        "The deployed model is a calibrated Linear SVM. "
        "Calibration enables probability estimates for "
        "confidence reporting and explainability."
    )


    # ======================================================
    # CLAIM ANALYSIS
    # ======================================================

    st.divider()

    st.subheader(
        "🧠 Claim Analysis"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            "**Claim Type**"
        )

        st.info(
            claim_info["claim_type"]
        )


    with col2:

        st.write(
            "**Entities**"
        )

        st.info(
            ", ".join(
                claim_info["entities"]
            )
            if claim_info["entities"]
            else "None"
        )


    with col3:

        st.write(
            "**Dates**"
        )

        st.info(
            ", ".join(
                claim_info["dates"]
            )
            if claim_info["dates"]
            else "None"
        )


    # ======================================================
    # SEARCH QUERIES
    # ======================================================

    with st.expander(
        "🔎 Search Queries Used"
    ):

        queries = retrieval.get(
            "queries",
            []
        )

        if queries:

            for query in queries:

                st.write(
                    "•",
                    query
                )

        else:

            st.write(
                "No search queries generated."
            )


    # ======================================================
    # EVIDENCE SUMMARY
    # ======================================================

    st.divider()

    st.subheader(
        "📚 Evidence Summary"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🟢 Supporting",
            aggregated["support_count"]
        )


    with col2:

        st.metric(
            "🔴 Refuting",
            aggregated["refute_count"]
        )


    with col3:

        st.metric(
            "⚪ Neutral",
            aggregated["neutral_count"]
        )


    # ======================================================
    # EVIDENCE DETAILS
    # ======================================================

    st.subheader(
        "🔍 Evidence Details"
    )


    if not verified_results:

        st.warning(
            "No sufficiently reliable evidence was found."
        )

    else:

        for index, evidence in enumerate(
            verified_results,
            start=1
        ):

            title = evidence.get(
                "title",
                "Untitled source"
            )

            label = evidence.get(
                "label",
                "NEUTRAL"
            )

            confidence = float(
                evidence.get(
                    "confidence",
                    0.0
                )
            )

            evidence_score = float(
                evidence.get(
                    "evidence_score",
                    0.0
                )
            )


            with st.expander(
                f"{index}. {title}"
            ):

                st.write(
                    "**NLI:**",
                    label
                )

                st.write(
                    "**NLI Confidence:**",
                    f"{confidence:.2%}"
                )

                st.write(
                    "**Evidence Score:**",
                    f"{evidence_score:.3f}"
                )

                st.write(
                    "**Source Quality:**",
                    f"{evidence.get('source_quality', 0.0):.2f}"
                )

                st.write(
                    evidence.get(
                        "content",
                        "No content available."
                    )
                )


                url = evidence.get(
                    "url",
                    ""
                )


                if url:

                    st.markdown(
                        f"[🌐 Open Source]({url})"
                    )


    # ======================================================
    # EXPLAINABLE AI
    # ======================================================

    st.divider()

    st.subheader(
        "💡 Explainable AI"
    )


    st.write(
        f"**ML Prediction:** "
        f"{xai_result['prediction']}"
    )


    st.write(
        f"**ML Confidence:** "
        f"{xai_result['confidence']:.2%}"
    )


    st.write(
        "The chart below shows which words influenced "
        "the ML prediction."
    )


    # ------------------------------------------------------
    # PREPARE XAI DATA
    # ------------------------------------------------------

    xai_features = xai_result[
        "features"
    ]


    xai_chart_data = pd.DataFrame(
        [
            {
                "Word": str(
                    feature["word"]
                ),
                "Weight": float(
                    feature["weight"]
                ),
            }

            for feature in xai_features
        ]
    )


    # ------------------------------------------------------
    # XAI BAR CHART
    # ------------------------------------------------------

    if not xai_chart_data.empty:

        st.bar_chart(
            xai_chart_data.set_index(
                "Word"
            ),
            use_container_width=True,
            height=400,
        )


    # ------------------------------------------------------
    # XAI DETAILS
    # ------------------------------------------------------

    with st.expander(
        "📖 View Detailed XAI Feature Weights"
    ):

        for feature in xai_features:

            word = str(
                feature["word"]
            )

            weight = float(
                feature["weight"]
            )


            if weight > 0:

                direction = "REAL"

            elif weight < 0:

                direction = "FAKE"

            else:

                direction = "NEUTRAL"


            st.write(
                f"**{word}** → "
                f"{weight:+.4f} "
                f"({direction})"
            )


    st.caption(
        "Positive weights support the REAL prediction, "
        "while negative weights support the FAKE prediction."
    )