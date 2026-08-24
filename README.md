# Household Energy Consumption Prediction

## Overview
This project focuses on predicting household electricity consumption using deep learning models for sequential data.
The project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository, containing approximately 2 million observations of household electricity measurements recorded at one-minute intervals.

Three recurrent neural network architectures were implemented and compared:

- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Recurrent Neural Network (RNN)

The prediction target is `Global_active_power`, representing the household's global active power consumption.
This project was developed as part of an Advanced Machine Learning course project.

---

## Objectives
- Analyze household electricity consumption patterns.
- Clean and preprocess raw electricity consumption data.
- Explore relationships and patterns among electricity-related variables.
- Perform feature engineering to derive additional information from the raw data.
- Develop LSTM, GRU, and RNN models for electricity consumption prediction.
- Compare model performance using regression metrics.
- Analyze the prediction results to identify the best-performing model.

---

## Dataset

The project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.
The dataset contains **2,075,259 observations and 9 variables**, representing household electricity measurements recorded at one-minute intervals.

### Variables
| Variable | Description |
|---|---|
| `Date` | Date of the measurement |
| `Time` | Time of the measurement |
| `Global_active_power` | Global active power consumed per minute |
| `Global_reactive_power` | Global reactive power consumed per minute |
| `Voltage` | Voltage measurement |
| `Global_intensity` | Global intensity measurement |
| `Sub_metering_1` | Energy sub-metering for the kitchen area |
| `Sub_metering_2` | Energy sub-metering for the laundry area |
| `Sub_metering_3` | Energy sub-metering for water heater and air conditioning |

### Target Variable

The prediction target is:

`Global_active_power`

The remaining variables are used as input features.

### Dataset Source
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The raw dataset is included in the `data/` directory.

---

## Data Preparation

### 1. Date and Time Processing
The `Date` and `Time` columns were combined into a single `Datetime` column to represent the observations as time-based data.
The resulting datetime variable was used to support further time-based analysis.

### 2. Data Type Conversion
Several numerical variables were initially stored as object/string values.
The numerical columns were converted into floating-point values to allow numerical analysis and modeling.

### 3. Missing Value Handling
Missing values were identified across the numerical electricity-related variables.
A total of **25,979 missing values** were identified in:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

The `Datetime` column did not contain missing values.
Missing values were handled before proceeding to subsequent analysis and modeling.

### 4. Outlier Analysis
Outliers were examined across the numerical variables to identify unusual electricity consumption observations.
The analysis was performed before normalization and model development.

### 5. Normalization
The numerical variables were normalized using `MinMaxScaler`.
This step scales the numerical features into a common range and prepares the data for neural network training.

---

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the characteristics and patterns of household electricity consumption.

The analysis included:
- Distribution analysis
- Consumption patterns
- Outlier analysis
- Correlation analysis
- Daily energy consumption
- Hourly energy consumption
- Energy usage by sub-metering area

### Daily Energy Consumption
The analysis showed that **Other Active Energy** contributed substantially more to daily consumption compared with the other categories analyzed.

### Energy Usage by Area
Among the three sub-metering areas, the **kitchen area showed the highest energy usage ratio** in the analysis.

### Hourly Consumption
The analysis of average energy consumption by hour was used to identify periods with higher electricity usage.
In the project analysis, consumption increased across the observed early-hour intervals and reached its highest value around **4 AM**.

---

## Feature Engineering
Feature engineering was performed to derive additional information from the electricity consumption variables.
The project explored:
- Active energy consumed by other equipment
- Daily energy consumption
- Energy usage ratios
- Energy consumption by hour
- Energy usage by area
- Correlation between electricity-related variables

These derived features were used to better understand household electricity consumption patterns and provide additional information for the modeling stage.

---

## Train-Test Split
The processed data was divided into training and testing sets using an **80:20 split**.
| Dataset | Observations |
|---|---:|
| Training | 1,639,424 |
| Testing | 409,856 |

The target variable was `Global_active_power`, while the remaining variables were used as input features.
The split was performed using `random_state=42` to ensure reproducibility.

---

## Model Development
Three recurrent neural network architectures were developed:

### LSTM
Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed to capture dependencies in sequential data.
LSTM uses memory cells and gating mechanisms to retain relevant information across sequential observations.

### GRU
Gated Recurrent Unit (GRU) is a recurrent neural network architecture with a simpler gating structure than LSTM.
GRU was implemented to evaluate whether a lighter recurrent architecture could achieve competitive performance on the household electricity dataset.

### RNN
Recurrent Neural Network (RNN) was implemented as a baseline recurrent architecture.
RNN processes sequential information by passing information from previous observations to subsequent computations.

---

## Model Architecture and Training
The models were implemented using TensorFlow.
The input data was transformed into a **3-dimensional format** to make it compatible with sequential neural network layers.
Each model consists of:
- One recurrent layer: LSTM, GRU, or RNN
- One Dense output layer

The models were compiled using:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 10 |
| Batch Size | 32 |

After training, each model was used to generate predictions on the testing data.

---

## Model Evaluation
The models were evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

### Results
| Model | MSE | MAE | R² Score |
|---|---:|---:|---:|
| LSTM | High | High | -118.83 |
| GRU | ~0.002 | ~0.0116 | 0.9632 |
| RNN | ~0.002 | ~0.0116 | 0.9629 |

### Performance Comparison
The results show a substantial difference between the three models.
**GRU achieved the best overall performance**, with an R² score of **0.9632**, slightly outperforming RNN with an R² score of **0.9629**.
Both GRU and RNN achieved much better results than LSTM under the experimental configuration.
The LSTM model produced an R² score of **-118.83**, indicating that the model failed to capture the target pattern effectively in this experiment.

---

## Training Performance
Training and validation performance were analyzed using:
- Training Loss vs. Validation Loss
- Training MAE vs. Validation MAE

### LSTM
The LSTM model showed fluctuations during training. The validation loss began increasing while training loss continued decreasing around the fourth epoch, indicating potential overfitting.
The model ultimately produced poor test performance, reflected by its negative R² score.

### GRU
GRU showed relatively stable training behavior, with decreasing loss and MAE throughout training.
The validation performance also remained relatively stable, indicating better generalization compared with LSTM.

### RNN
RNN demonstrated relatively stable training behavior and achieved performance close to GRU.
However, GRU achieved a slightly higher R² score.

---

## Prediction Results
After training, the three models were used to predict `Global_active_power` on the test data.
Each model generated the **first 10 prediction values** from the test set for comparison.
The predictions represent estimated global active power consumption.
