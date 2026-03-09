# LLM License Plate Evaluation Framework

> **NLI-based contradiction detection, zero-shot re-evaluation, toxicity scoring, bias analysis, and visual reporting for LLM classification outputs.**

A rigorous post-classification evaluation framework for the [LLM License Plate Moderation Pipeline](https://github.com/bushramaryam/llm-license-plate-moderation). Takes raw GPT classification results and puts them through three layers of evaluation — technical NLI metrics, visual data analysis, and a comprehensive quality report.

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | **99 annotated plates** |
| Contradictions detected | **24 inconsistent judgments** |
| NLI model | `facebook/bart-large-mnli` |
| Toxicity model | `unitary/toxic-bert` |
| Languages analyzed | **15** |
| Fairness metric | Cohen's Kappa |

---

## Two-Notebook Structure

This repo contains two notebooks that run **in order**:

```
Dataset.csv
    │
    ▼
[Notebook 1] LLMEvaluationMetrics.ipynb
    │   NLI scoring, zero-shot re-evaluation,
    │   toxicity detection, contradiction detection
    │
    ├──→ plates_evaluation_results.csv
    └──→ contradiction_analysis.csv
              │
              ▼
         AnnotatedDataset.csv (human-reviewed subset)
              │
              ▼
[Notebook 2] LicensePlateUSeCaseEvaluation.ipynb
        Data quality, class distribution,
        language bias, category consistency,
        outlier detection, quality report
```

---

## Notebook 1 — `LLMEvaluationMetrics.ipynb`

**Purpose:** Technical evaluation of GPT predictions using NLI and zero-shot models as independent judges.

### What it does:

| Step | Method | Detail |
|------|--------|--------|
| Load & normalize | pandas | Handles NaN, strips whitespace, capitalizes labels |
| Zero-shot re-evaluation | `facebook/bart-large-mnli` | Re-predicts Approve/Reject + category independently of GPT |
| NLI scoring | Cross-encoder NLI | Tests whether the plate description *entails*, *contradicts*, or is *neutral* to both the true and predicted labels |
| Toxicity scoring | `unitary/toxic-bert` | Scores plate descriptions for harmful content |
| Contradiction detection | Combined NLI + zero-shot | Flags cases where NLI and zero-shot disagree with GPT |
| Keyword matching | Rule-based | 7-category keyword dictionary as a third signal |
| Accuracy & confusion matrix | `sklearn` | Decision-level and category-level metrics |

### Output columns in `plates_evaluation_results.csv`:

| Column | Description |
|--------|-------------|
| `harm_score` | Toxicity probability (toxic-bert) |
| `consistency` | NLI label: entailment / neutral / contradiction |
| `consistency_score` | NLI confidence |
| `faithfulness` | NLI label for predicted hypothesis |
| `faithfulness_score` | NLI confidence for predicted |
| `hallucination` | Boolean — True if contradiction detected |
| `error_type` | Potential False Positive / False Negative / Correct |

### Sample contradiction from `contradiction_analysis.csv`:

```
plate_number:    rehmat20
true_decision:   Reject
pred_decision:   Approve          ← GPT disagreement
nli_true_label:  neutral (0.72)
nli_pred_label:  neutral (0.85)
zs_decision:     Approve (0.90)   ← zero-shot agrees with GPT
reason:          "Description neutral to TRUE hypothesis (rejected due to
                  missing/broken plate, 0.72) and neutral to PRED hypothesis
                  (compliant and approved, 0.85). Zero-shot leans Approve (0.90)."
```

---

## Notebook 2 — `LicensePlateUSeCaseEvaluation.ipynb`

**Purpose:** Visual and statistical analysis of the annotated dataset — bias, distribution, quality, and a full report.

### What it does:

| Section | Analysis |
|---------|---------|
| **Data Quality** | Null rates, duplicate plates, field completeness |
| **Class Distribution** | Approve vs Reject balance, category breakdown |
| **Language Bias** | Cohen's Kappa across 15 languages, per-language accuracy |
| **Category Consistency** | Cross-category confusion, mis-labelling patterns |
| **Outlier Detection** | Plates flagged by multiple evaluation signals |
| **Quality Report** | Auto-generated comprehensive summary |

### Languages analyzed:
Arabic, Urdu, French, Spanish, German, Chinese, Japanese, Hindi, Portuguese, Russian, Turkish, Korean, Italian, Dutch, English


---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| `facebook/bart-large-mnli` | NLI zero-shot classification |
| `unitary/toxic-bert` | Toxicity scoring |
| `transformers`, `torch` | Model inference |
| `langdetect` | Language identification |
| `sklearn` | Accuracy, confusion matrix, Cohen's Kappa |
| `pandas`, `numpy` | Data processing |
| `matplotlib`, `seaborn`, `plotly` | Visualization |
| `wordcloud`, `textstat` | Text analysis |

---

## Setup

```bash
git clone https://github.com/bushramaryam/llm-license-plate-evaluation
cd llm-license-plate-evaluation
pip install transformers torch langdetect scikit-learn pandas numpy \
            matplotlib seaborn plotly wordcloud textstat sentence-transformers
```

**Run order:**
1. `notebooks/01_LLMEvaluationMetrics.ipynb` — generates `plates_evaluation_results.csv` and `contradiction_analysis.csv`
2. `notebooks/02_LicensePlateUSeCaseEvaluation.ipynb` — reads both outputs for visual analysis

> GPU recommended for Notebook 1 (BART-large-MNLI + toxic-bert inference). Google Colab with T4 works well.

---

## 🔗 Related

**Pipeline repo**: [`llm-license-plate-moderation`](https://github.com/BushraMaryam/LLM-License-Plate-Moderation-System) — the GPT classification pipeline that produced the results this repo evaluates.

---

## 👩‍💻 Author

**Bushra Maryam** — AI & Backend Engineer  
[LinkedIn](https://www.linkedin.com/in/bushramaryam) · [GitHub](https://github.com/bushramaryam) · [Portfolio](https://bushramaryam.github.io)
