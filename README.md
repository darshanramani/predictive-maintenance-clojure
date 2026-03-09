# Predictive Maintenance using Clojure and Machine Learning

## Project Overview

This project implements a **machine learning system for predictive maintenance** using the **AI4I Predictive Maintenance dataset**.
The goal is to predict whether a machine will **fail or not** based on sensor data.

The project is implemented in **Clojure** and uses the **Weka machine learning library** to train and evaluate models.

---

# Dataset

Dataset used: **AI4I Predictive Maintenance Dataset**

Total records: **10,000**

Main features used for prediction:

* Air temperature (K)
* Process temperature (K)
* Rotational speed (rpm)
* Torque (Nm)
* Tool wear (minutes)
* Machine type

Target variable:

**Machine failure (0 = No failure, 1 = Failure)**

---

# Machine Learning Models Used

Two models were implemented and compared:

1. **Decision Tree (J48)**
2. **Random Forest**

These models were implemented using the **Weka Java Machine Learning library**, which works with Clojure because Clojure runs on the **Java Virtual Machine (JVM)**.

---

# Data Preprocessing

Before training the model, some preprocessing steps were performed:

* Removed unnecessary columns (IDs and failure type columns)
* Converted categorical variables to nominal format
* Converted the target variable to nominal format
* Selected the correct class attribute for prediction

---

# Model Evaluation

The models were evaluated using:

**10-fold Cross Validation**

This means the dataset is divided into 10 parts and the model is trained and tested 10 times to produce reliable results.

---

# Results

| Model         | Accuracy |
| ------------- | -------- |
| Decision Tree | 98.29%   |
| Random Forest | 98.64%   |

Random Forest performed slightly better than the Decision Tree.

A **confusion matrix** is also printed to show detailed prediction performance.

---

# Decision Tree Visualization

The project also displays the **Decision Tree structure in a GUI window** so the rules learned by the model can be visually inspected.

This helps understand how the model makes predictions based on sensor values.

---

# Project Structure

```
predictive-maintenance-clojure
│
├── data
│   └── ai4i2020.csv
│
├── src
│   └── maintenance
│       └── core.clj
│
├── project.clj
└── README.md
```

---

# Requirements

Before running the project, install:

* **Java (JDK 8 or later)**
* **Leiningen (Clojure build tool)**

Check installation:

```
java -version
lein --version
```

---

# How to Run the Project

Navigate to the project folder and run:

```
lein clean
lein run
```

If `lein` is not in PATH, run:

```
C:\Users\Admin\lein.bat run
```

---

# Program Output

The program will:

1. Load the dataset
2. Preprocess the data
3. Train the machine learning models
4. Evaluate the models using cross validation
5. Print accuracy and confusion matrices
6. Compare the models
7. Open a GUI window showing the Decision Tree structure

---

# Conclusion

This project demonstrates how machine learning can be applied to **predict machine failures using sensor data**.
Using the AI4I dataset, both Decision Tree and Random Forest models achieved high accuracy, with Random Forest performing slightly better.
