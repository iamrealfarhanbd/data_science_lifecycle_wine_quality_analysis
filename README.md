# Wine Quality Prediction Project

## Overview

This project uses machine learning to predict the quality of red wine based on its physicochemical properties. The code implements a Random Forest Regressor model to estimate wine quality scores from a dataset of wine attributes.

## Features

* Data loading from the UCI Machine Learning Repository
* Exploratory data analysis (EDA) with visualization
* Feature engineering (creation of an interaction term)
* Training a Random Forest regression model
* Hyperparameter tuning using GridSearchCV
* Model evaluation with RMSE and R-squared
* Saving the trained model

## Code Description

The `wine_quality.py` script contains the following sections:

1.  **Imports Libraries**: Imports necessary Python libraries.
2.  **Data Acquisition**: Loads the wine quality dataset.
3.  **Data Cleaning and Preprocessing**: Handles missing values and duplicates.
4.  **Exploratory Data Analysis (EDA)**: Visualizes data characteristics.
5.  **Feature Engineering**: Creates new features.
6.  **Data Preparation for Modeling**: Splits data into training and testing sets.
7.  **Model Development**: Trains a Random Forest model.
8.  **Hyperparameter Tuning**: Optimizes model parameters.
9.  **Visualizing the Results**: Plots actual vs. predicted values.
10. **Saving the Model**: Saves the trained model for later use.

## How to Run the Code

1.  **Prerequisites**:
    * Python 3.x
    * Required Python libraries: pandas, scikit-learn, matplotlib, seaborn, joblib.  Install them using `pip`:
        ```bash
        pip install pandas scikit-learn matplotlib seaborn joblib
        ```
2.  **Run the script**:
    * Save the code as `wine_quality.py`.
    * Open a terminal or command prompt in the same directory.
    * Execute the script:
        ```bash
        python wine_quality.py
        ```

##  Important Considerations
    * The script assumes the data file is available at the specified URL.
    * The Random Forest model can be further improved by experimenting with different hyperparameters and feature engineering techniques.
## License
This project is licensed under the MIT License. See the LICENSE file for details.
## Acknowledgments
* UCI Machine Learning Repository for the wine quality dataset.
* Scikit-learn documentation for machine learning algorithms and tools.
* Matplotlib and Seaborn for data visualization.
* Joblib for model serialization.
* Pandas for data manipulation and analysis.
* NumPy for numerical operations.
* Seaborn for enhanced data visualization.
* Matplotlib for plotting.
* Joblib for model serialization.
* Scikit-learn for machine learning algorithms.
* Pandas for data manipulation.
* NumPy for numerical operations.
* Seaborn for enhanced data visualization.
* Matplotlib for plotting.
* Joblib for model serialization.
* Scikit-learn for machine learning algorithms.
* Pandas for data manipulation.
* NumPy for numerical operations.
* Seaborn for enhanced data visualization.
* Matplotlib for plotting.
* Joblib for model serialization.
* Scikit-learn for machine learning algorithms.

##  Author
\[Farhan Ahmed\]
##  Date
\[2025-05-09\]
##  Version
\[1.0\]
##  Contact
For any questions or feedback, please contact the author at [Iamfarhan09@gmail.com](mailto:iamfarhan09@gmail.com).
