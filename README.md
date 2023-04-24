# CE888---Stress-Forecasting---University-trial

[Link to download the datasets](https://github.com/italha-d/Stress-Predict-Dataset)

[Link to instruction](https://moodle.essex.ac.uk/pluginfile.php/1007595/course/section/139943/2022_CE888_Assignment_2.pdf?time=1675939526588)

Firstly, there are two files: "Assignment2_CE888.ipynb" and "Sensor.py" should be downloaded and put in the same folder before running.




1. The notebook ("Assignment2_CE888.ipynb") contains the code sections for illustrative examples and data exploration on Jupyter Notebook.
These examples include:
- The plots of time series data ("BVP", "HR", "EDA", "TEMP") for person 1
- Timestamp tags exploration: indicate the difference in labeling "stress" and "no stress" between Person 1 and the others.
- Model prediction for each person.

2. The notebook ("Sensor.py") contains all the defined function and the main code including:
- Define a function to load and process raw data of volunteers and then combine all together.

  **wrangle(self, filepath, s)**
  
  Filepath is the link to download the datasets from Github:  "https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/"
  
  s: the number from 1 to 35 represents for the volunteers.
  
- Define a function to create a lag feature

  **def create_data(self, lag_length=1)**
  
  lag_length: the fixed time period.

