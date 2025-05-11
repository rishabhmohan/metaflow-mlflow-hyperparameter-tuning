from metaflow import FlowSpec, step, Parameter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import itertools
import pandas as pd
import mlflow
import mlflow.sklearn


class HyperparameterTuningFlow(FlowSpec):
    # Define hyperparameter ranges in the code
    random_state = Parameter('random_state', default=42)

    # Define hyperparameters for different models
    rf_params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10, None]
    }

    lr_params = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }

    svc_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    @step
    def start(self):
        # Load the Iris dataset
        self.data = load_iris()

        # Start a parent MLflow run
        self.parent_run = mlflow.start_run()
        self.parent_run_id = self.parent_run.info.run_id

        self.next(self.split_data)

    @step
    def split_data(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.data, self.data.target, test_size=0.2, random_state=self.random_state
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.next(self.generate_hyperparameters)

    @step
    def generate_hyperparameters(self):
        # Generate all combinations of hyperparameters for each model
        self.hyperparameter_combinations = []

        # RandomForest
        rf_combinations = list(itertools.product(self.rf_params['n_estimators'], self.rf_params['max_depth']))
        self.hyperparameter_combinations.extend([('RandomForest', params) for params in rf_combinations])

        # LogisticRegression
        lr_combinations = list(itertools.product(self.lr_params['C'], self.lr_params['solver']))
        self.hyperparameter_combinations.extend([('LogisticRegression', params) for params in lr_combinations])

        # SVC
        svc_combinations = list(itertools.product(self.svc_params['C'], self.svc_params['kernel']))
        self.hyperparameter_combinations.extend([('SVC', params) for params in svc_combinations])

        self.next(self.train_model, foreach='hyperparameter_combinations')

    @step
    def train_model(self):
        # Unpack the model type and hyperparameters for this iteration
        model_type, params = self.input

        if model_type == 'RandomForest':
            n_estimators, max_depth = params
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                random_state=self.random_state)

        elif model_type == 'LogisticRegression':
            C, solver = params
            self.model = LogisticRegression(C=C, solver=solver, random_state=self.random_state, max_iter=200)

        elif model_type == 'SVC':
            C, kernel = params
            self.model = SVC(C=C, kernel=kernel, random_state=self.random_state)

            # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Log the child run under the parent run
        with mlflow.start_run(run_id=self.parent_run_id, nested=True) as child_run:
            self.child_run_id = child_run.info.run_id

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        # Evaluate the model on the test set
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"{self.input[0]} model accuracy with hyperparameters {self.input[1]}: {self.accuracy:.4f}")

        # Log parameters and metrics to MLflow using a unique run ID
        with mlflow.start_run(nested=True) as eval_run:
            params_dict = dict(zip(['param1', 'param2'], self.input[1]))
            params_dict["model_type"] = self.input[0]
            mlflow.log_params(params_dict)
            mlflow.log_metric("accuracy", self.accuracy)

            # Log the model to MLflow with input example
            input_example = self.X_test[0:1]
            mlflow.sklearn.log_model(self.model, "model", input_example=input_example)

        self.model_type = self.input[0]
        self.params = self.input[1]
        self.next(self.gather_results)

    @step
    def gather_results(self):
        # Gather results from all parallel steps
        self.next(self.join)

    @step
    def join(self, inputs):
        # Gather results from all iterations
        self.results = [(input.model_type, input.params, input.accuracy) for input in inputs]

        # Find the best hyperparameters for each model
        model_results = {}
        for model_type, params, accuracy in self.results:
            if model_type not in model_results:
                model_results[model_type] = []
            model_results[model_type].append((params, accuracy))

            # Find the best hyperparameters for each model type
        self.best_hyperparameters = {}
        for model_type, results in model_results.items():
            best_params, best_accuracy = max(results, key=lambda x: x[1])
            self.best_hyperparameters[model_type] = (best_params, best_accuracy)
            print(f"Best hyperparameters for {model_type}: {best_params} with accuracy: {best_accuracy:.4f}")

            # Save results to a CSV file
        all_results = [(model_type, str(params), accuracy) for model_type, params, accuracy in self.results]
        df = pd.DataFrame(all_results, columns=['Model', 'Hyperparameters', 'Accuracy'])
        df.to_csv('hyperparameter_tuning_results.csv', index=False)

        self.next(self.end)

    @step
    def end(self):
        # End the parent MLflow run
        mlflow.end_run()

        print("Hyperparameter tuning completed.")
        print("Results have been saved to hyperparameter_tuning_results.csv")


if __name__ == "__main__":
    HyperparameterTuningFlow()