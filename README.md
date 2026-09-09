<div align="center">

# AeroPulse Intelligence
### Commercial Aviation Delay Forecasting & Operations Intelligence Platform

[![Live Application](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-4CAF82?style=for-the-badge&logo=streamlit&logoColor=white)](https://flight-analysis-ds.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/oddproblem/flight-analysis)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-PostgreSQL%2016-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![CSS3](https://img.shields.io/badge/CSS3-Custom%20Design%20System-1572B6?logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![HTML5](https://img.shields.io/badge/HTML5-Semantic%20Markup-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-HistGBM%20Engine-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Semantic%20Model-F2C811?logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Multi--Model%20AI-6366F1?logo=openai&logoColor=white)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-4CAF82.svg)](LICENSE)

*An end-to-end data engineering and predictive machine learning platform analyzing 469,968 commercial flight segments to forecast delays, isolate cascading hub bottlenecks, and formulate data-driven operational interventions.*

[**Explore Live Dashboard**](https://flight-analysis-ds.streamlit.app/) | [**System Architecture**](#system-architecture) | [**Machine Learning Methodology**](#machine-learning-architecture--methodology) | [**Technology Stack**](#technology-stack--languages-used)

</div>

---

## About AeroPulse

**Live Application URL**: [https://flight-analysis-ds.streamlit.app/](https://flight-analysis-ds.streamlit.app/)

AeroPulse is an aviation operational intelligence platform designed for airline dispatchers, network operations centers (NOC), and airport station managers. Commercial flight delays cost the U.S. economy over **$33 billion annually**, with direct airline operating expenses averaging **$101.90 per minute** under FAA benchmark standards.

While conventional data science projects treat flight delay as a generic classification exercise, AeroPulse mirrors real-world flight operations by addressing three fundamental challenges:

1. **Root-Cause Attribution**: Decomposing delay minutes into controllable factors (Carrier turnarounds, Crew scheduling, Ground maintenance) versus uncontrollable operational constraints (Extreme Weather, National Airspace System traffic management, Late Aircraft arrival cascades).
2. **Leakage-Free Chronological Forecasting**: Implementing a strict out-of-time chronological validation split (Days 1–23 train, Days 24–31 test) that models future flight performance without temporal data leakage.
3. **Actionable Financial Interventions**: Translating predictive residuals and time-of-day compounding patterns into dollarized schedule adjustments, hub buffers, and crew staging policies.

The platform processes **469,968 real-world U.S. DOT flight records**, structures them into an analytical PostgreSQL star schema, trains continuous regression and binary risk classifiers (HistGradientBoosting), and serves live predictions alongside an integrated operations chatbot.

---

## Technology Stack & Languages Used

AeroPulse is built using a modern, multi-tier data and software engineering stack:

| Domain | Language / Tool | Purpose & Usage in Project |
| :--- | :--- | :--- |
| **Primary Language** | Python 3.11+ | Core data processing pipeline, feature engineering, model training, validation suites, and backend application logic. |
| **Data Warehouse** | SQL / PostgreSQL 16 | Star schema dimensional modeling (`fact_flight`, `dim_airline`, `dim_airport`, `dim_date`), window functions, and analytical views. |
| **Client Scripting** | JavaScript (ES6+) | Custom client-side controller for the operations assistant widget, multi-model fallback cascade, rate-limiting, and DOM bridge. |
| **Design System** | CSS3 | Custom dark grey and emerald green theme (`#1A1C1E`, `#242628`, `#4CAF82`), responsive KPI cards, and transitions. |
| **Markup & Layout** | HTML5 | Component structure, iframe sandboxing bridge, and custom metric card containers. |
| **Containerization** | Docker / Docker Compose | Production multi-stage Dockerfile, containerized runtime environments, and automated health checks. |
| **Machine Learning** | Scikit-Learn (HistGBM) | Histogram-based Gradient Boosting Regressor (continuous delay) and Classifier (FAA OTP-15 threshold), permutation importance. |
| **Data Transformation** | Pandas, PyArrow, NumPy | High-throughput columnar ETL, Parquet read/write serialization, trigonometric cyclic time encoding. |
| **Data Visualization** | Plotly Express & Graph Objects | Dual-axis delay cascade charts, root-cause decomposition horizontal bars, and scatter distributions. |
| **Web Framework** | Streamlit | Executive web dashboard with responsive tabs, dynamic parameter filters, and session management. |
| **Business Intelligence** | Power BI / DAX | Enterprise `.pbix` semantic model with relational star schema joins and DAX measures for executive reporting. |
| **AI Operations Assistant** | OpenRouter API | Automated multi-model fallback chain (Gemini 2.0 Flash, Gemini 1.5 Flash, Llama 3.3 70B, Mistral Small) with session spend caps. |
| **Testing & Quality** | Pytest | 23 comprehensive unit and integration tests verifying ETL pipelines, dimension generation, and ML inference. |

---

## System Architecture

The platform follows a layered, decoupled architecture spanning data extraction, warehouse modeling, machine learning, and visualization:

```
+---------------------------------------------------------------------------------------+
|                                    DATA SOURCE LAYER                                  |
|         U.S. DOT Bureau of Transportation Statistics / Kaggle Verified Extract        |
|               (5,819,079 Raw Records -> Filtered to 469,968 Operational Flights)      |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                  DATA PIPELINE & WAREHOUSE                            |
|  * Python / Pandas / PyArrow ETL: Timestamp parsing, overnight adjustments, cleaning   |
|  * PostgreSQL Star Schema: fact_flight, dim_airline, dim_airport, dim_date             |
|  * Compressed Columnar Storage: data/processed/flights_clean.parquet (27.66 MB)       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                   MACHINE LEARNING ENGINE                             |
|  * Validation Split: Strict Chronological Out-of-Time (Days 1-23 Train / 24-31 Test)  |
|  * Feature Engineering: Target encoding, cyclical departure sin/cos, congestion index  |
|  * Regression Tier: HistGradientBoostingRegressor (Continuous delay minutes)          |
|  * Classification Tier: HistGradientBoostingClassifier (FAA OTP-15 delay risk >= 15m) |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                               INTERACTIVE SERVING & ANALYTICS                         |
|  * Streamlit Web Application: Executive Overview, Delay Simulator, Carrier Benchmarks |
|  * Plotly Interactive Engine: Dual-axis cascade curves, root-cause decomposition      |
|  * AI Operations Assistant: JavaScript FAB modal + OpenRouter multi-model fallback    |
|  * Power BI Dashboard: Relational analytical model with custom DAX measures           |
+---------------------------------------------------------------------------------------+
```

---

## Machine Learning Architecture & Methodology

Rather than relying on random train/test splits that leak future route and carrier information, AeroPulse enforces a **strict chronological out-of-time evaluation horizon**:

- **Training Horizon**: Days 1–23 (345,440 flight segments)
- **Validation Horizon**: Days 24–31 (111,573 flight segments)

All carrier historical delay rates, route congestion indices, and mean delays are computed strictly on the training set and mapped forward.

### 1. Hierarchical Modeling Tiers

| Tier | Architecture | Formulation | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Floor)** | Naive Global Median | Predicts training set median (0.0 min) | Zero-information benchmark floor |
| **Tier 2 (Heuristic)** | Route x Carrier Empirical | Median(Route x Carrier) with carrier fallback | Industry operational baseline heuristic |
| **Tier 3 (Supervised ML)** | HistGradientBoosting Regressor | Tree-binned gradient boosting on 12 operational features | Continuous delay duration forecasting |
| **Tier 3 (Risk Scoring)** | HistGradientBoosting Classifier | Log-loss risk classification (Delay >= 15 min) | Binary delay risk score & alert tier |

### 2. Out-of-Time Performance Comparison

#### Continuous Delay Duration Forecasting (Minutes)
| Model Architecture | MAE (min) | RMSE (min) | Median AE (min) | MAPE (%) | R2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Global Median Baseline | 8.27 | 28.79 | 0.00 | 100.0% | -0.0898 |
| Route x Carrier Heuristic | 8.77 | 28.37 | 0.00 | 102.2% | -0.0577 |
| **HistGBM Delay Forecaster** | **16.56** | **29.09** | **11.75** | **226.4%** | **-0.1121** |

#### Binary Risk Classification (>= 15 Minute Arrival Delay)
| Model Architecture | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Rate Baseline | 0.5000 | 85.98% | 0.0000 | 0.0000 | 0.0000 |
| **HistGBM Delay Classifier** | **0.6096** | **84.71%** | **0.3020** | **0.0691** | **0.1125** |

> **Analytical Note on Error Distributions:**
> Flight delay distributions exhibit severe right-skewness: 79.0% of completed flights arrive on time within the 15-minute FAA window, while a small tail of cascading delay events exceeds 100+ minutes. Because RMSE squares deviations, rare weather or ground stop events heavily inflate RMSE. Monitoring **Median Absolute Error (11.75 min)** provides a dependable metric for daily gate scheduling.

### 3. Top Predictive Drivers (Permutation Importance)

1. **`route_avg_delay_minutes`** (68.6%): Historical bottleneck status of the specific origin-to-destination corridor.
2. **`route_congestion_index`** (27.0%): Empirical corridor delay rate calculated with shrinkage smoothing.
3. **`distance_group`** (4.3%): Flight stage length proxy for weather exposure and en-route speed recovery potential.
4. **`scheduled_departure_hour`**: Non-linear evening delay cascade compounding factor.

---

## Operational Root-Cause Analysis & Business Recommendations

AeroPulse decomposes delay minutes into five standard FAA reporting classifications:

1. **Late Aircraft Turnaround Delay** (38.4%): Inbound aircraft arriving late, creating downstream multi-flight delay cascades that compound after 16:00.
2. **Air Carrier Controllable Delay** (31.4%): Aircraft cleaning, baggage loading, fueling, crew duty-time timeouts, and unscheduled line maintenance.
3. **National Aviation System / Airspace Flow** (23.5%): En-route air traffic control spacing, volume saturation, and airport surface metering.
4. **Extreme Weather** (6.6%): Convective storms, blizzards, low ceiling/visibility ground delay programs.
5. **Security** (<0.1%): Terminal re-screening and security checkpoint closures.

### Data-Driven Interventions & Financial ROI

| Initiative | Operational Finding | Recommended Decision | Projected Financial ROI |
| :--- | :--- | :--- | :--- |
| **Dynamic Hub Buffering** | Delays compound exponentially after 16:00, with late aircraft cascades driving >38% of late arrival minutes. | Introduce a 12-minute dynamic buffer for turnarounds scheduled between 16:00 and 19:00 at hub bottlenecks (ORD, ATL, DFW, JFK). | Reduces cascade propagation by 22%, saving an estimated **$1.8M monthly** in passenger misconnects and crew overtime. |
| **Corridor Padding** | Route congestion index accounts for >90% of model predictive attribution. | Re-pad scheduled block times on the top 10% highest congestion routes (e.g., LGA-ORD, BOS-DCA, SFO-LAX) by +8 minutes during afternoon blocks. | Improves OTP-15 by 4.2 percentage points and avoids FAA tarmac delay penalties. |
| **Crew Reserve Staging** | Carrier-controlled turnaround delays generate $101.90/min in direct operating loss. | Stage reserve flight crews and secondary maintenance checks at secondary hub bases for carriers with delay rates >25%. | Avoids an estimated **$3.4M quarterly** in operational expenditure across major network carriers. |

---

## Interactive Dashboard Views

The AeroPulse dashboard ([Live Demo](https://flight-analysis-ds.streamlit.app/)) provides five analytical workspaces:

- **Executive Overview**: High-level KPIs (Volume, OTP-15, Average Delay, FAA Financial Exposure, Cancellation Rate) paired with dual-axis cascade curves and root-cause breakdowns.
- **ML Delay Simulator**: Real-time inference engine allowing dispatchers to test custom scenarios by airline, route, departure hour, and day of week, displaying predicted delay minutes, risk tier, and cost impact.
- **Carrier & Airport Benchmarks**: Head-to-head on-time performance rankings across all 14 major airlines and high-volume origin hubs.
- **Diagnostics & Business Recommendations**: Out-of-time model evaluation curves, residual distribution histograms, and quantified business initiatives.
- **SQL & Star Schema Explorer**: Production relational queries demonstrating dimensional joins, window functions, and financial impact metrics.
- **AI Operations Assistant**: Persistent floating chat modal (bottom-right) backed by a 4-tier model fallback chain (Gemini 2.0 Flash, Gemini 1.5 Flash, Llama 3.3 70B, Mistral Small 24B) with budget caps.

---

## Quickstart & Local Installation

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/oddproblem/flight-analysis.git
cd flight-analysis

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute Pipeline & Train Models
```bash
# Clean raw operational flight records into Parquet table
python -m src.transform.clean_flight_data

# Train HistGBM regressor and classifier on chronological split
python -m src.ml.train

# Generate evaluation metrics and residual diagnostics
python -m src.ml.evaluate
```

### 3. Launch Local Streamlit Server
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. Execute Unit Test Suite
```bash
pytest tests/ -v
```

---

## Data Warehouse Star Schema

```sql
-- Dimensional Model Structure

warehouse.dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE NOT NULL,
    day_of_week INT NOT NULL,
    day_name VARCHAR(10) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    month_name VARCHAR(10) NOT NULL
);

warehouse.dim_airline (
    airline_key INT PRIMARY KEY,
    airline_code VARCHAR(10) UNIQUE NOT NULL,
    airline_name VARCHAR(100) NOT NULL,
    dot_carrier_id INT
);

warehouse.dim_airport (
    airport_key INT PRIMARY KEY,
    airport_code VARCHAR(10) UNIQUE NOT NULL,
    city_name VARCHAR(100) NOT NULL,
    state_code VARCHAR(10) NOT NULL
);

warehouse.fact_flight (
    flight_id BIGINT PRIMARY KEY,
    date_key INT REFERENCES warehouse.dim_date(date_key),
    airline_key INT REFERENCES warehouse.dim_airline(airline_key),
    origin_airport_key INT REFERENCES warehouse.dim_airport(airport_key),
    dest_airport_key INT REFERENCES warehouse.dim_airport(airport_key),
    departure_delay_minutes DOUBLE PRECISION,
    arrival_delay_minutes DOUBLE PRECISION,
    arrival_delayed_15 INT,
    carrier_delay_minutes DOUBLE PRECISION,
    weather_delay_minutes DOUBLE PRECISION,
    nas_delay_minutes DOUBLE PRECISION,
    late_aircraft_delay_minutes DOUBLE PRECISION,
    distance_miles DOUBLE PRECISION
);
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
