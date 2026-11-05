import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_NAME = "facebook/bart-large-mnli"

_tokenizer = None
_model = None


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def get_nli_model():

    global _tokenizer, _model

    if _tokenizer is None or _model is None:

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        _model.eval()

    return _tokenizer, _model


# ---------------------------------------------------------
# TEXT PROCESSING
# ---------------------------------------------------------

def tokenize_words(text):

    return set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text.lower()
        )
    )


def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# RELEVANCE
# ---------------------------------------------------------

def relevance_score(claim, sentence):

    claim_words = tokenize_words(claim)

    sentence_words = tokenize_words(sentence)

    if not claim_words:
        return 0.0

    overlap = (
        claim_words.intersection(
            sentence_words
        )
    )

    return (
        len(overlap)
        / len(claim_words)
    )


def select_relevant_sentences(
    claim,
    evidence,
    max_sentences=3,
    min_relevance=0.20,
):

    sentences = split_sentences(
        evidence
    )

    scored = []

    for sentence in sentences:

        score = relevance_score(
            claim,
            sentence
        )

        if score >= min_relevance:

            scored.append(
                (
                    score,
                    sentence
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        sentence
        for score, sentence in scored[:max_sentences]
    ]


# ---------------------------------------------------------
# RAW BART NLI
# ---------------------------------------------------------

def run_nli(
    claim,
    evidence,
):

    tokenizer, model = get_nli_model()

    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    scores = {}

    for index, probability in enumerate(
        probabilities
    ):

        label = model.config.id2label[
            index
        ].lower()

        scores[label] = float(
            probability
        )

    return {
        "SUPPORTS": scores.get(
            "entailment",
            0.0
        ),

        "REFUTES": scores.get(
            "contradiction",
            0.0
        ),

        "NEUTRAL": scores.get(
            "neutral",
            0.0
        ),
    }


# ---------------------------------------------------------
# MAIN VERIFICATION
# ---------------------------------------------------------

def verify_claim(
    claim: str,
    evidence: str,
):

    if not claim or not claim.strip():

        raise ValueError(
            "Claim cannot be empty."
        )

    if not evidence or not evidence.strip():

        raise ValueError(
            "Evidence cannot be empty."
        )

    claim = claim.strip()

    evidence = evidence.strip()


    # -----------------------------------------------------
    # FIND RELEVANT EVIDENCE SENTENCES
    # -----------------------------------------------------

    relevant_sentences = select_relevant_sentences(
        claim,
        evidence,
        max_sentences=3,
        min_relevance=0.20,
    )


    # No relevant sentence means we cannot verify it.
    if not relevant_sentences:

        return {
            "label": "NEUTRAL",
            "confidence": 1.0,
            "relevance": 0.0,

            "scores": {
                "SUPPORTS": 0.0,
                "REFUTES": 0.0,
                "NEUTRAL": 1.0,
            },

            "relevant_sentences": [],
        }


    # -----------------------------------------------------
    # RUN NLI ON EACH RELEVANT SENTENCE
    # -----------------------------------------------------

    results = []

    for sentence in relevant_sentences:

        relevance = relevance_score(
            claim,
            sentence
        )

        nli_scores = run_nli(
            claim,
            sentence
        )

        results.append(
            {
                "sentence": sentence,
                "relevance": relevance,
                "scores": nli_scores,
            }
        )


    # -----------------------------------------------------
    # SELECT BEST SUPPORT / REFUTE / NEUTRAL
    # -----------------------------------------------------

    support_score = max(
        item["scores"]["SUPPORTS"]
        for item in results
    )

    refute_score = max(
        item["scores"]["REFUTES"]
        for item in results
    )

    neutral_score = max(
        item["scores"]["NEUTRAL"]
        for item in results
    )


    score_map = {
        "SUPPORTS": support_score,
        "REFUTES": refute_score,
        "NEUTRAL": neutral_score,
    }


    label, confidence = max(
        score_map.items(),
        key=lambda item: item[1]
    )


    # -----------------------------------------------------
    # BEST RELEVANCE
    # -----------------------------------------------------

    best_relevance = max(
        item["relevance"]
        for item in results
    )


    # -----------------------------------------------------
    # LOW RELEVANCE = UNRELATED EVIDENCE
    # -----------------------------------------------------

    if best_relevance < 0.35:

        label = "NEUTRAL"

        confidence = 1.0


    # -----------------------------------------------------
    # LOW NLI CONFIDENCE = UNCERTAIN
    # -----------------------------------------------------

    elif confidence < 0.70:

        label = "NEUTRAL"


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "label": label,

        "confidence": float(
            confidence
        ),

        "relevance": float(
            best_relevance
        ),

        "scores": {
            "SUPPORTS": float(
                support_score
            ),

            "REFUTES": float(
                refute_score
            ),

            "NEUTRAL": float(
                neutral_score
            ),
        },

        "relevant_sentences": [
            item["sentence"]
            for item in results
        ],
    }