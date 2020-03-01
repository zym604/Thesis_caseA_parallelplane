from autoPyTorch import AutoNetClassification
# Other imports for later usage
import openml
import json
autonet = AutoNetClassification(config_preset="tiny_cs", result_logger_dir="logs/")
# Get the current configuration as dict
current_configuration = autonet.get_current_autonet_config()
hyperparameter_search_space = autonet.get_hyperparameter_search_space()

task = openml.tasks.get_task(task_id=31)
X, y = task.get_X_and_y()
ind_train, ind_test = task.get_train_test_split_indices()
X_train, Y_train = X[ind_train], y[ind_train]
X_test, Y_test = X[ind_test], y[ind_test]
autonet = AutoNetClassification(config_preset="tiny_cs", result_logger_dir="logs/")
# Fit (note that the settings are for demonstration, you might need larger budgets)
results_fit = autonet.fit(X_train=X_train,
                          Y_train=Y_train,
                          validation_split=0.3,
                          max_runtime=300,
                          min_budget=60,
                          max_budget=100,
                          refit=True)
# Save fit results as json
with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
# See how the random configuration performs (often it just predicts 0)
score = autonet.score(X_test=X_test, Y_test=Y_test)
pred = autonet.predict(X=X_test)
print("Model prediction:", pred[0:10])
print("Accuracy score", score)
pytorch_model = autonet.get_pytorch_model()
print(pytorch_model)
# Load fit results as json
with open("logs/results_fit.json") as file:
    results_fit = json.load(file)
# Create an autonet
autonet_config = {
    "result_logger_dir" : "logs/",
    "budget_type" : "epochs",
    "log_level" : "info", 
    "use_tensorboard_logger" : True,
    "validation_split" : 0.0
    }
autonet = AutoNetClassification(**autonet_config)
# Sample a random hyperparameter configuration as an example
hyperparameter_config = autonet.get_hyperparameter_search_space().sample_configuration().get_dictionary()
# Refit with sampled hyperparameter config for 120 s. This time on the full dataset.
results_refit = autonet.refit(X_train=X_train,
                              Y_train=Y_train,
                              X_valid=None,
                              Y_valid=None,
                              hyperparameter_config=results_fit['optimized_hyperparameter_config'],
                              autonet_config=autonet.get_current_autonet_config(),
                              budget=1000)
# Save json
with open("logs/results_refit.json", "w") as file:
    json.dump(results_refit, file)
# See how the random configuration performs (often it just predicts 0)
score = autonet.score(X_test=X_test, Y_test=Y_test)
pred = autonet.predict(X=X_test)
print("Model prediction:", pred[0:10])
print("Accuracy score", score)
pytorch_model = autonet.get_pytorch_model()
print(pytorch_model)
