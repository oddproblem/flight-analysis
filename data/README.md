# AeroPulse Real-World Flight Data Source

## Primary Data Source
This project utilizes the **U.S. Department of Transportation (USDOT) / Bureau of Transportation Statistics (BTS)** flight reliability and delay dataset sourced from Kaggle.

- **Kaggle Dataset**: [2015 Flight Delays and Cancellations (USDOT)](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Raw File Path**: `data/raw/flights.csv` (592 MB, 5,819,079 flight segments)
- **Reporting Scope**: All major U.S. commercial carriers and 300+ commercial airports across the United States.

## Cleaned Operational Dataset
The automated pipeline in `src/transform/clean_flight_data.py` ingests the raw records and outputs production Parquet files:
- `data/processed/flights_clean.parquet` (469,968 flight segments for Month 1, 27.66 MB compressed with Snappy)

### Preserved Features & Breakdown:
- Carrier delay (`carrier_delay_minutes`)
- Weather delay (`weather_delay_minutes`)
- National Airspace System delay (`national_air_system_delay_minutes`)
- Security delay (`security_delay_minutes`)
- Late aircraft delay (`late_aircraft_delay_minutes`)
- On-time performance flag (OTP-15, arrival delay < 15 min)
- Continuous delay duration in minutes (positive and signed)
- Route code, scheduled departure hour block, and distance group
