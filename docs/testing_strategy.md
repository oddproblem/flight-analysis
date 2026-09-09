# Testing Notes

I did not start this project with a full test suite. At first I mainly checked the January dataset by running each script and comparing the output. After changing the transformation and loader a few times, I added automated tests for the parts that were easiest to break.

## Unit tests

`tests/test_clean_transform_helpers.py` checks the transformation rules I rely on most:

- HHMM parsing, including `2400` and invalid times
- route creation
- calendar fields derived from `flight_date`
- completed, cancelled, and diverted status logic
- rejection of rows marked both cancelled and diverted
- arrival on-time logic
- delay-cause totals
- duplicate removal
- required source and warehouse fields

`tests/test_load_helpers.py` checks the loader logic that can be tested without a database:

- date dimension creation
- airline attribute conflicts
- airport dimension creation
- required values before loading
- surrogate-key mapping for fact rows

`tests/test_ml_pipeline.py` verifies the machine learning and inference components:

- Feature engineering transforms (cyclic hour encodings, Bayesian shrinkage route congestion)
- FlightDelayPredictor inference contract and risk tier thresholds
- Output schema consistency (probabilities, estimated delay minutes, cost exposure)

## PostgreSQL integration test


The database part needs a real PostgreSQL instance, so GitHub Actions starts PostgreSQL 18 and runs `tests/test_postgres_integration.py`.

The test uses a small fake flight dataset, creates the real warehouse schema and analytics views, runs the loader, and checks that the fact table and reporting views return the expected results.

I skip this test during a normal local `pytest` run because I do not always have the test database running. It is enabled when:

```text
RUN_POSTGRES_TESTS=1
```

is set.

## Validation scripts

There are separate validators for the raw CSV, the cleaned Parquet file, and the PostgreSQL warehouse.

One issue I fixed while reviewing the project was that a failed validation could still finish with process exit code `0`. That looks fine when running the file manually, but CI or a scheduled job would treat it as success. Critical failures now raise an exception so the process actually fails.

## Commands I use locally

```powershell
python -m pytest -q
python -m compileall -q src
```

When I want to run the database integration test locally, I start PostgreSQL first and set `RUN_POSTGRES_TESTS=1`.
