# Decision Trees and Random Forests

## Objective
The objective of this project is to implement and compare tree-based machine learning models, namely Decision Tree and Random Forest, for classification using a real-world dataset.

---

## Tools and Technologies
Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  

---

## Dataset
Heart Disease Dataset (heart.csv)

The dataset contains medical attributes such as age, sex, chest pain type, cholesterol, maximum heart rate, and other health-related features used to predict heart disease.

---

## Workflow

### Data Preprocessing
Loaded the dataset using pandas  
Checked and handled missing values  
Separated features and target variable (target)

### Feature Scaling
Applied StandardScaler to normalize the feature values

### Model Training

Decision Tree  
Trained using DecisionTreeClassifier  
Controlled overfitting using max_depth  
Visualized the tree structure  

Random Forest  
Trained using RandomForestClassifier  
Improved performance using multiple decision trees  

### Evaluation
Compared accuracy of both models  
Used cross-validation to validate model performance  

### Feature Importance
Analyzed which features contribute most to predictions  

---

## Results
Decision Tree Accuracy is good  
Random Forest Accuracy is higher and more stable  
Cross-validation confirms consistent performance  

---

## Output

Decision Tree Visualization  
![Decision Tree](tree_output.jpg)

Feature Importance  
![Feature Importance](feature_importance.jpg)

---

## Key Insights
Random Forest performs better than Decision Tree  
Important features include cp, ca, thal, and thalach  
Limiting tree depth helps reduce overfitting  

---

## How to Run

Install required libraries  
pip install pandas numpy matplotlib scikit-learn  

Run the program  
python decision_tree_random_forest.py  

---

## Project Structure

Decision-Trees-and-Random-Forests/

decision_tree_random_forest.py  
heart.csv  
tree_output.jpg  
feature_importance.jpg  
README.md  

---

## Conclusion
Successfully implemented Decision Tree and Random Forest models, compared their performance, and analyzed feature importance. Random Forest provided better accuracy and generalization.

---

## Key Learning
Tree-based models are effective for classification  
Random Forest reduces overfitting  
Feature importance improves interpretability  
Cross-validation ensures reliable evaluation  
