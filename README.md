# Student Score Predictor 📚

This is a small ML project I made to predict a student's final score based on a few basic factors like study hours attendance and previous marks

## What I used

- Python
- Pandas
- Scikit learn
- Streamlit

## How it works

I used a Random Forest Regression model for the prediction. The dataset is split into training and testing data, and the model is trained using the training part.

The Streamlit interface is used to take inputs from the user and display the prediction.

## How to Run
First install the required libraries:

```bash
pip install -r requirements.txt

## Run the project

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Project structure

```text
student_performance_predictor
 app.py
 student_data.csv
 requirements.txt
 README.md
 .gitignore
```

## Dataset

The dataset is **sample data created for this project** not real student data. It is just for learning and testing!!


## Future improvements

- Use a larger real world dataset with appropriate permissions
- Try other regression models and compare their results
- Add more student related features
- Add visual analysis of the dataset

## Limitations

The current dataset is small and synthetic so the model cannot be used to make reliable predictions about actual students.

## Author

**Mahika Reddy**

B.Tech Information Technology student exploring Python, machine learning and software projects.
