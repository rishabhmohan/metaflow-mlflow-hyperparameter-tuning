# Hyperparameter Tuning Flow  
  
This project implements a hyperparameter tuning workflow for machine learning models using Metaflow and MLflow. The workflow trains and evaluates different models with various hyperparameters and logs the results to MLflow.  
  
## Prerequisites  
  
- Python 3.8 or higher  
- [Anaconda](https://www.anaconda.com/products/distribution) (recommended for managing environments)  
  
## Setup  
  
### Step 1: Create and Activate a Virtual Environment  
  
It is recommended to create a virtual environment to manage dependencies. You can use `conda` to create and activate a virtual environment.  
  
```
conda create -n hyperparameter-tuning python=3.8  
conda activate hyperparameter-tuning
```
 

### Step 2: Install Required Packages
 
Install the necessary packages using pip.

```
pip install metaflow  
pip install mlflow  
pip install scikit-learn  
 ```
 

### Step 3: Run the Workflow
 
Run the Metaflow script to start the hyperparameter tuning workflow.


```python hyper_flow.py run  ```
 

### Step 4: View Results in MLflow
  Start the MLflow UI to view the logged results:

```mlflow ui```
  Open a web browser and go to http://127.0.0.1:5000. You will be able to see the experiment runs, parameters, metrics, and models logged by MLflow. After running the workflow, you can view the best hyperparameters and their corresponding accuracy for each model in the MLflow UI and the hyperparameter_tuning_results.csv file.

 Best hyperparameters for RandomForest: (100, 10) with accuracy: 0.9333  
 Best hyperparameters for LogisticRegression: (1, 'lbfgs') with accuracy: 0.9000  
 Best hyperparameters for SVC: (1, 'rbf') with accuracy: 0.9667  

### Step 5: Build AND RUN the Docker image
```
docker build -t fastapi-prediction-app .  
docker run -p 80:80 fastapi-prediction-app  
```

### Serve the model
Now, when you open http://localhost:80 in your web browser, you should see separate input fields for each feature.
Upon submission, the form will send the data to the /predict/ endpoint and return the prediction.


