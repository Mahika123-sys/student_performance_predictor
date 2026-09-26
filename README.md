# Student Score Predictor 📚

A small machine learning project that estimates a student's final score from three inputs:

- Study hours per day
- Attendance percentage
- Previous score

I made this project to understand the basic workflow of a machine learning application: preparing data, training a model, checking its performance, and connecting the model to a simple user interface.

## What I used

- Python
- Pandas
- Scikit-learn
- Streamlit

## How it works

The dataset contains sample student records. I used **Random Forest Regression** because it is easy to apply to this type of small tabular dataset and can also provide feature-importance values.

The data is split into training and testing sets. After training, the app calculates:

- **MAE (Mean Absolute Error)** to show the average prediction error
- **R² score** to give a basic measure of how well the model fits the test data

The Streamlit interface then takes new student details and sends them to the trained model.

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
student-performance-predictor/
│
├── app.py
├── student_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The dataset is **synthetic**, meaning it was created for this project rather than collected from real students. It is only intended for learning and demonstration.

## Future improvements

- Use a larger real-world dataset with appropriate permissions
- Try other regression models and compare their results
- Add more student-related features
- Add visual analysis of the dataset
- Deploy the Streamlit app online

## Limitations

The current dataset is small and synthetic, so the model cannot be used to make reliable predictions about actual students.

## Author

**Mahika Reddy**

B.Tech Information Technology student exploring Python, machine learning and software projects.
