# Malaria Outbreak Prediction Model

A machine learning system that predicts malaria outbreaks using environmental, healthcare, and socioeconomic factors through Random Forest classification.

## Project Structure

```
MALARIA/
├── .scannerwork/
├── venv/
├── .coverage
├── cleaning.py
├── confusion_matrix.png
├── coverage.xml
├── dim_dates.csv
├── dim_demographics.csv
├── dim_environment.csv
├── dim_health_initiatives.csv
├── dim_healthcare.csv
├── dim_infrastructure.csv
├── dim_location.csv
├── dim_prevention.csv
├── dim_socioeconomic.csv
├── dim_weather.csv
├── fact_malaria_cases.csv
├── feature_importance.csv
├── feature_importance.png
├── loading.py
├── model.py
├── processed_data.csv
└── test_malaria_db.py
```

## Overview

This project implements a predictive model for malaria outbreak detection using machine learning techniques. The system analyzes multiple dimensions of data including:
- Environmental factors
- Demographic information
- Healthcare accessibility
- Infrastructure data
- Prevention measures
- Socioeconomic indicators
- Weather patterns

## Data Sources

The project uses a dimensional data model with the following key files:
- **Fact Table**: fact_malaria_cases.csv - Contains the main malaria case records
- **Dimension Tables**:
  - dim_dates.csv - Temporal dimensions
  - dim_demographics.csv - Population demographics
  - dim_environment.csv - Environmental factors
  - dim_health_initiatives.csv - Health programs and initiatives
  - dim_healthcare.csv - Healthcare facility information
  - dim_infrastructure.csv - Infrastructure availability
  - dim_location.csv - Geographical information
  - dim_prevention.csv - Malaria prevention measures
  - dim_socioeconomic.csv - Socioeconomic indicators
  - dim_weather.csv - Weather-related data

## Key Components

### Data Processing
- **cleaning.py**: Handles data cleaning and preprocessing
- **loading.py**: Manages data loading and ETL processes

### Model
- **model.py**: Contains the Random Forest implementation
- **test_malaria_db.py**: Test suite for database operations

### Analysis Outputs
- **confusion_matrix.png**: Visual representation of model performance
- **feature_importance.csv**: Detailed feature importance scores
- **feature_importance.png**: Visual representation of feature importance
- **processed_data.csv**: Final processed dataset used for modeling

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Cleaning:
```bash
python cleaning.py
```

2. Load and Process Data:
```bash
python loading.py
```

3. Train and Run Model:
```bash
python model.py
```

4. Run Tests:
```bash
python -m pytest test_malaria_db.py
```

## Model Evaluation

Model performance metrics and visualizations can be found in:
- **confusion_matrix.png** - For model accuracy evaluation
- **feature_importance.png** - For understanding feature significance

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project uses dimensional modeling for data organization. Make sure all dimension tables are properly loaded before running the fact table integration.

