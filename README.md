# AI Data Analytics Platform

An intelligent data analytics assistant. Upload a CSV, talk to
it in natural language, get auto-generated insights, anomaly detection,
forecasts and a professional PDF report — all powered by a **local LLM via
Ollama**. No API keys. No cloud calls.

---

## 1. Tech Stack

| Layer            | Tool                                |
| ---------------- | ----------------------------------- |
| UI               | Streamlit                           |
| LLM (local)      | Ollama running Llama 3 or Qwen 3    |
| Data wrangling   | Pandas, NumPy                       |
| Visualization    | Plotly                              |
| ML               | scikit-learn (IsolationForest, RF)  |
| Reports          | ReportLab + kaleido                 |

---

## 2. Folder Structure

```
project/
├── app.py                       # Streamlit entry point
├── requirements.txt
├── README.md
├── data/
│   ├── sample_sales.csv         # bundled demo dataset
│   └── _generate_sample.py      # script to regenerate it
├── reports/                     # generated PDF reports land here
├── assets/                      # logos / images (optional)
└── modules/
    ├── __init__.py
    ├── data_loader.py           # CSV load + type inference
    ├── analyzer.py              # profiling + Ollama wrapper
    ├── chart_generator.py       # heuristic chart recommender
    ├── insight_engine.py        # rule-based insights + LLM polish
    ├── anomaly_detector.py      # IsolationForest outliers
    ├── forecasting.py           # Linear / RandomForest forecasts
    ├── chat_agent.py            # NL ↔ dataframe Q&A
    └── report_generator.py      # ReportLab PDF builder
```

---


## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI (app.py)                 │
│  Upload │ Overview │ Insights │ Chat │ Anomalies │ Forecast │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────────────┐
        ▼                  ▼                          ▼
┌────────────────┐ ┌────────────────┐       ┌────────────────────┐
│  data_loader   │ │ chart_generator│       │   chat_agent       │
│ (CSV + types)  │ │ (Plotly recs)  │       │ (intents + LLM)    │
└────────────────┘ └────────────────┘       └─────────┬──────────┘
        │                  ▲                          │
        ▼                  │                          ▼
┌────────────────┐ ┌────────────────┐       ┌────────────────────┐
│   analyzer     │ │ insight_engine │       │  Ollama HTTP API   │
│ (profile+LLM)  │ │ (rules + LLM)  │◀──────│ http://localhost:  │
└────────┬───────┘ └────────┬───────┘       │      11434         │
         │                  │               └────────────────────┘
         ▼                  ▼
┌────────────────┐ ┌────────────────┐
│ anomaly_detect │ │  forecasting   │
│ (IsoForest)    │ │ (LR / RF)      │
└────────┬───────┘ └────────┬───────┘
         └────────┬─────────┘
                  ▼
         ┌────────────────────┐
         │  report_generator  │
         │  (ReportLab PDF)   │
         └────────────────────┘
```

---

## 4. Data Flow Diagram

```
 CSV file ─▶ data_loader.load_csv ─▶ pandas.DataFrame ─┐
                                                       │
                            ┌──────────────────────────┼──────────────────────────┐
                            ▼                          ▼                          ▼
                  data_loader.basic_overview  analyzer.dataset_profile  chart_generator.recommend_charts
                            │                          │                          │
                            ▼                          ▼                          ▼
                   insight_engine.generate_insights ◀──┘                  Plotly figures
                            │
                            ▼
                   ask_ollama (HTTP POST /api/generate)
                            │
                            ▼
                   narrated executive summary
                            │
   user question ──▶ chat_agent.answer ──▶ deterministic intent OR Ollama fallback
                            │
                            ▼
                   anomaly_detector.detect_anomalies (IsolationForest)
                            │
                            ▼
                   forecasting.forecast (RF / LR + 95% interval)
                            │
                            ▼
                   report_generator.build_report ──▶ reports/report_<ts>.pdf
```

---

## 5. Sample Dataset

The bundled `data/sample_sales.csv` contains 1,500 synthetic sales orders
with the columns:

```
OrderDate, Region, Category, Channel, Product, Units,
UnitPrice, Discount, Revenue, Profit, CustomerSatisfaction
```

It deliberately includes:

* A few extreme outliers (for the anomaly module to find).
* Some missing values in `CustomerSatisfaction`.
* A mild upward trend in monthly revenue (for forecasting).

Regenerate it any time with:

```bash
python data/_generate_sample.py
```
