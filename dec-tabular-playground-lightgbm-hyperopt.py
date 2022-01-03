# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 3.388813, "end_time": "2021-12-13T19:45:34.665415", "exception": false, "start_time": "2021-12-13T19:45:31.276602", "status": "completed"} tags=[]
import math
import lightgbm
from lightgbm.basic import LightGBMError
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from scipy import stats, special, optimize
import hyperopt
from functools import partial

# %% papermill={"duration": 0.023493, "end_time": "2021-12-13T19:45:34.705510", "exception": false, "start_time": "2021-12-13T19:45:34.682017", "status": "completed"} tags=[]
print(f"Using LightGBM version {lightgbm.__version__}")

# %% papermill={"duration": 0.021589, "end_time": "2021-12-13T19:45:34.742619", "exception": false, "start_time": "2021-12-13T19:45:34.721030", "status": "completed"} tags=[]
TRAINING = False

# %% [markdown] papermill={"duration": 0.015156, "end_time": "2021-12-13T19:45:34.773315", "exception": false, "start_time": "2021-12-13T19:45:34.758159", "status": "completed"} tags=[]
# ## GPU support

# %% [markdown] papermill={"duration": 0.015172, "end_time": "2021-12-13T19:45:34.803925", "exception": false, "start_time": "2021-12-13T19:45:34.788753", "status": "completed"} tags=[]
# We need to do a bit of work to get LightGBM to work with GPU - this is based on [this](https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm) notebook by Vinh Nguyen.

# %% papermill={"duration": 36.540795, "end_time": "2021-12-13T19:46:11.360134", "exception": false, "start_time": "2021-12-13T19:45:34.819339", "status": "completed"} tags=[]
# step 1) remove the current installation and install a fresh one from source
# !rm -r /opt/conda/lib/python3.7/site-packages/lightgbm
# !git clone --recursive https://github.com/Microsoft/LightGBM

# %% papermill={"duration": 48.79793, "end_time": "2021-12-13T19:47:00.266187", "exception": false, "start_time": "2021-12-13T19:46:11.468257", "status": "completed"} tags=[]
# step 2) install boost dev library
# !apt-get install -y -qq libboost-all-dev

# %% papermill={"duration": 192.489154, "end_time": "2021-12-13T19:50:12.929169", "exception": false, "start_time": "2021-12-13T19:47:00.440015", "status": "completed"} tags=[] language="bash"
# cd LightGBM
# rm -r build
# mkdir build
# cd build
# cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
# make -j$(nproc)

# %% papermill={"duration": 1.854717, "end_time": "2021-12-13T19:50:14.954188", "exception": false, "start_time": "2021-12-13T19:50:13.099471", "status": "completed"} tags=[]
# step 4) reinstall using our new, GPU-enabled build
# !cd LightGBM/python-package/;python3 setup.py install --precompile

# %% papermill={"duration": 1.645463, "end_time": "2021-12-13T19:50:16.770958", "exception": false, "start_time": "2021-12-13T19:50:15.125495", "status": "completed"} tags=[]
# step 5) do a small bit of post processing and clean up folder
# !mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# !rm -r LightGBM

# %% [markdown] papermill={"duration": 0.169647, "end_time": "2021-12-13T19:50:17.111373", "exception": false, "start_time": "2021-12-13T19:50:16.941726", "status": "completed"} tags=[]
# ## Setting up 

# %% papermill={"duration": 21.373379, "end_time": "2021-12-13T19:50:38.654618", "exception": false, "start_time": "2021-12-13T19:50:17.281239", "status": "completed"} tags=[]
train_data_raw = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
train_data_raw = train_data_raw[~(train_data_raw["Cover_Type"] == 5)]  # only 1 instance
train_data_raw = train_data_raw.drop(["Soil_Type7", "Soil_Type15", "Id"], axis=1)  # no instances 
test_data_raw = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
test_data_raw = test_data_raw.drop(["Soil_Type7", "Soil_Type15"], axis=1)

# %% papermill={"duration": 0.177861, "end_time": "2021-12-13T19:50:39.003829", "exception": false, "start_time": "2021-12-13T19:50:38.825968", "status": "completed"} tags=[]
cts_columns = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
soil_cols = [f"Soil_Type{i}" for i in range(1, 41) if i not in [7, 15]]
wilderness_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]
target = "Cover_Type"


# %% papermill={"duration": 0.182335, "end_time": "2021-12-13T19:50:39.355239", "exception": false, "start_time": "2021-12-13T19:50:39.172904", "status": "completed"} tags=[]
def get_trained_models(train_data,
                       create_model,
                       n_models: int = 5):
    """
    create n models trained through k-fold
    cross validation
    """
    all_scores = []
    all_models = []
    kf = KFold(n_splits=5)
    for train_index, val_index in kf.split(train_data):

        # split up our data
        train_x = train_data.iloc[train_index, :].drop(target, axis=1).values.astype(np.float32)
        train_y = train_data.iloc[train_index, :][target].values.astype(np.float32)
        val_x = train_data.iloc[val_index, :].drop(target, axis=1).values.astype(np.float32)
        val_y = train_data.iloc[val_index, :][target].values.astype(np.float32)

        # fit our model
        model = create_model()
        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        acc = accuracy_score(pred, val_y)

        # store and print result
        all_scores.append(acc)
        all_models.append(model)
        print(f"Score for fold {len(all_scores)} is {acc}")

    print(f"Mean score is {np.mean(all_scores)}")
    return all_models, all_scores
    

def create_submission(test_data, 
                      all_models):
    """
    make predictions using each of our model
    and take the modal class for each sample
    """
    all_predictions = []
    for model in all_models:
        pred = model.predict(test_data.drop("Id", axis=1))
        all_predictions.append(pred)

    all_pred = np.stack(all_predictions).T
    submission = test_data[["Id"]]
    submission["Cover_Type"] = stats.mode(all_pred, axis=1).mode
    submission["Cover_Type"] = submission["Cover_Type"].astype("int") 
    return submission



# %% [markdown] papermill={"duration": 0.168589, "end_time": "2021-12-13T19:50:39.693150", "exception": false, "start_time": "2021-12-13T19:50:39.524561", "status": "completed"} tags=[]
# ## Feature engineering

# %% papermill={"duration": 3.259954, "end_time": "2021-12-13T19:50:43.124609", "exception": false, "start_time": "2021-12-13T19:50:39.864655", "status": "completed"} tags=[]
def feature_engineer(data):
    
    # 1) add count cols for soil and wilderness
    data["soil_count"] = data[soil_cols].sum()
    data["wilderness_count"] = data[wilderness_cols].sum()
    
    # 2) fix our aspect
    data["Aspect"][data["Aspect"] < 0] += 360
    data["Aspect"][data["Aspect"] > 359] -= 360
    
    # 3) fix the hillshade columns
    data["Hillshade_9am"] = data["Hillshade_9am"].clip(0, 255)
    data["Hillshade_Noon"] = data["Hillshade_Noon"].clip(0, 255)
    data["Hillshade_3pm"] = data["Hillshade_3pm"].clip(0, 255)
    
    # 4) add a few other features that others have found useful
    data['Highwater'] = (data["Vertical_Distance_To_Hydrology"] < 0).astype(int)
    data['Euclidean_Distance_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology'] ** 2 + data['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    data['Manhattan_Distance_to_Hydrolody'] = data['Horizontal_Distance_To_Hydrology'] + data['Vertical_Distance_To_Hydrology']
    
    return data

train_data = feature_engineer(train_data_raw)
test_data = feature_engineer(test_data_raw)


# %% [markdown] papermill={"duration": 0.170489, "end_time": "2021-12-13T19:50:43.466155", "exception": false, "start_time": "2021-12-13T19:50:43.295666", "status": "completed"} tags=[]
# ## LightGBM

# %% [markdown] papermill={"duration": 0.171056, "end_time": "2021-12-13T19:50:43.806693", "exception": false, "start_time": "2021-12-13T19:50:43.635637", "status": "completed"} tags=[]
# While various neural network architectures have achieved state-of-the-art performance on tasks involving data with some degree of temporal/spatial invariance (eg images, time series data, audio, NLP), Kaggle leaderboards can attest to the power of gradient boosted tree models when working with homogenous, tabular data like we are here. This section sets out briefly what a gradient boosted tree model is, and some of the special features of LightGBM.
#
# ### Gradient boosted trees
#
# Boosting is a type of ensemble approach, in which we sequentially combine "weak" (ie high bias, low variance) to create one much more powerful learner. We can take this approach with many different kinds of learners, although here we will only consider tree-based models. At a high level, our approach is as follows:
#
# 1. Start with an initial (tree) model  $f_0$, which is a single decision tree, which minimizes our loss function $\sum_{i=1}^nL(\hat{y^{(i)}}, y^{(i)})$
# 2. Fit a new model $f_1$ to the *pseudo-residuals*, defined as $r_i = -[\frac{\partial L(y^{(i)}, z)}{\partial z}]|_{z=\hat{y^{(i)}}}$ 
# 3. Add this tree to our model so that our new model is $f_0(x) + \alpha f_1(x)$, where alpha is our learning rate
# 4. Repeat steps 2 and 3 to sequentially build up our model, until we have a final model which is a linear combination of N trees
#
# Our prediction is then the sum of the predictions from each tree (weighted by the alphas), although depending on the target variable we may also apply a sigmoid or softmax function. Note that where we are predicting a continuous variable with the mean squared error loss function, then our pseudo-residuals are just our residuals. However, the advantage of using the pseudoresiduals is that this extends our approach to other differentiable loss functions.  In effect, we are doing gradient descent in function space in order to minimise our loss function.
#
# ### LightGBM
#
# LightGBM is a framework for building gradient boosted tree models developed by Microsoft. Like other popular frameworks like XGBoost and CatBoost, it contains a large number of optimizations to ensure that it can be trained efficiently on large datasets. In particular, three optimizations of note implemented in LightGBM are:
#
# 1. **Gradient One Sided Sampling** - the most time consuming part of the algorithm is building the individual trees, and within this finding the optimal split points for each split. One optimization implemented in most frameworks is to quantize continuous features, so that we split the feature into bins and then only need to check splitting between bins rather than at each individual value. However, the process of quantizing the data itself can be time consuming, so in LightGBM we downsample the available data when doing this to make it faster. In particular, we do so in such a way that we oversample instances with large gradients - intuitively, the idea is that these are instances which are currently poorly categorized by our model, and so it will pay to construct find the optimal split points for our tree with these in mind.
# 2. **Exclusive Feature Bundling** - for some datasets we can have very sparse data, and many features that do not take non-zero values simultaneously. This is the case with the soil type and wilderness area columns that we have here. LightGBM will look to see if it can combine some of these into single features, which again reduces the time taken in training the individual trees (as there are fewer features to consider for each split).
# 3. **Optimal Category Encoding** - it is commonplace to one-hot encode categorical features. However, for features with high cardinality this doesn't work well for tree-based learners as splitting on any such encoded variable is likely not to be particularly informative (as there may be a small number of observations for any one value). A better approach is to split categorical feature into two groups based on its value - LightGBM provides an efficient way to do this. 
#
# LightGBM also includes numerous other optimizations, including support for GPU-based training, parallelization of the training of individual trees where possible and support for sparse data storage. 
#
# ### Hyperparameters
#
# The LightGBM classifier model exposes a large number of hyperparameters which govern how the model is built and trained. Some of the most crucial hyperparameters are:
#
# 1. **num_leaves** - this controls the number of leaves in each decision tree learner. Note that unlike some other frameworks, LightGBM will build trees in a leaf-based rather than depth-based manner. This means at each stage it will add a node to the leaf where it thinks the greatest reduction in loss can be achieved, not necessarily to the shallowest leaf. This means we can end up with unbalanced trees, but tends to lead to faster convergence of the algorithm.
# 2. **max_depth** - as the name suggests, the maximum depth of any given decision tree learner
# 3. **min_data_in_leaf** - this parameter controls (again as the name would suggest) the minimum amount of observations allowable in a leaf for a split to be valid. This can help to prevent the tree overfitting by creating splits which are very specific to certain examples
# 4. **n_estimators** - the total number of decision trees that are built (ie the number of boosting iterations)
# 5. **learning_rate** - as per the formula in the "Gradient boosted trees" section above, this controls the $\alpha$ factor that we multiply predictions from successive decision tree learners by in our formula 
# 6. **lambda_l1** - LightGBM supports two forms of regularization. The first penalizes our model by the number of nodes (essentially L1 regularization), which is controlled by this parameter
# 7. **lambda_l2** - the second regularization parameter, this penalizes our model based on the sum of the square of the weigths on for each leaf
#

# %% papermill={"duration": 0.179057, "end_time": "2021-12-13T19:50:44.158717", "exception": false, "start_time": "2021-12-13T19:50:43.979660", "status": "completed"} tags=[]
def get_lightgbm_model(params = None):
    
    if not params:
        params = {"boosting_type": "goss",
                  "num_classes": 6,
                  "num_leaves": 10,
                  "max_depth": 4,
                  "min_data_in_leaf": 10,
                  "verbose": 1,
                  "max_bin": 63}
    model = lightgbm.LGBMClassifier(**params)
    return model



# %% papermill={"duration": 0.180262, "end_time": "2021-12-13T19:50:44.508923", "exception": false, "start_time": "2021-12-13T19:50:44.328661", "status": "completed"} tags=[]
if TRAINING:
    model = get_lightgbm_model() 
    x_data = train_data.drop(target, axis=1).values.astype(np.float32)
    y_data = train_data[target].values.astype(np.float32)
    log_eval = lightgbm.log_evaluation(period=1, show_stdv=True)
    model.fit(x_data, y_data, callbacks=[log_eval])
    pred = model.predict(x_data)
    accuracy_score(y_data, pred)


# %% [markdown] papermill={"duration": 0.170738, "end_time": "2021-12-13T19:50:44.849249", "exception": false, "start_time": "2021-12-13T19:50:44.678511", "status": "completed"} tags=[]
# ## Hyperparameter tuning

# %% [markdown] papermill={"duration": 0.16992, "end_time": "2021-12-13T19:50:45.190295", "exception": false, "start_time": "2021-12-13T19:50:45.020375", "status": "completed"} tags=[]
# We are now ready to start tuning our hyperparameters. Hyperparameter optimisation seeks to find the best values for parameters of the model that cannot be learned directly from the data during training, such as those highlighted in the previous section. We are going to use the Hyperopt library - this allows us to do something called *Bayesian Hyperparameter Tuning*.
#
# ### Non-bayesian hyperparameter tuning
#
# There are several common methods of hyperparameter tuning:
#
# 1. **Manual** - one option is just to run models against validation datasets with a bunch of different sets of parameter values and see what works best. This is simple, but requires you to make decisions about the best values to try, is very manual, and can quickly become very cumbersome for more sophisticated models like LightGBM.
# 2. **Grid Search** - a more systematic approach is to define as set of values for each hyperparameter (our grid), and then systematically run each possible combination. This is no longer a manual process, but we still need to make our own judgement on exactly which values to try. Further, there is something of a curse of dimensionality here - adding one more value can mean a large number of additional combinations to check if we have lots of different parameters. We are therefore also called on to decide a priori which hyperparameters are most important and hence deserve the most different values to try.
# 3. **Random Search** - an alternative is to define a *distribution* for each variable instead. This allows us to embed knowledge about which values may be most suitable for each parameter, but is less prescriptive than the grid search. Also, importantly, we no longer have to make a call on which are the most important parameters to focus on or run the risk of trying lots of combinations with some totally unsuitable value for one of our parameters.
#
#
# ### Bayesian hyperparamter tuning
#
# An issue with all the above is that we don't retain any knowledge of previous sets of hyperparameters that we have tried. Training our model for a given set of hyperparameters is a hugely expensive operation. If we can use our knowledge of previous sets of hyperparameters we have tried to inform the next set we try and this means we have to try fewer sets and hence train fewer models then this could be much more efficient - this is the core idea of Bayesian hyperparameter tuning.
#
# The hyperopt library supports several algorithms for doing this, but the only one I have used here is *Tree Parzen Estimators*. This described very well in [this](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) blog post, but a rough outline is as follows:
#
# 1. We define a set of hyperparameters that we would like to explore, with a distribution for each - this is the "tree" in the name, as we can have some hyperparameters which will depend on others, or whose use is conditional on the value of others.
# 2. Next we need to define a critera for evaluating a proposed set of hyperparameters we might like evaluate - generally this is the *Expected Improvement*, defined as: 
# $$\begin{aligned}
# IE_{y^*}(x) = \int_{-\inf}^{y*}(y^* - y)P(y|x) dy
# \end{aligned}$$
# Intuitively, we want a value $x$ where the expected reduction in our loss score $y$ is big.
# 3. The next task is to get a formula for $P(y|x)$ wherein we can make use of the previous combinations we have tried.  In Tree Parzen Estimators, we instead model 
#     $$ P(x|y) =   \left\{
# \begin{array}{ll}
#       l(x) \quad \text{if} \space y \lt y^* \\
#       g(x) \quad \text{if} \space y \geq y^* \\
# \end{array} 
# \right.  $$
# Note that we have two distributions - one for when our loss score is lower than the target $y^*$, and one when it is higher. If we apply Bayes law and rearrange we arrive at the following formula for our Expected Improvement:
# $$
# P(y|x) = \frac{\gamma y^* l(x) - l(x)\int_{-\inf}^{y^*}P(y)dy}{\gamma l(x) + (1 - \gamma)g(x)}
# $$
# We see that to minimize this, we would like to take an x such that we minimze $\frac{l(x)}{g(x)}$ - this makes intuitive sense, as this will have us pick a value of x where the score is likely to be lower than our target $y^*$, and unlikely to be higher than our target.  
# 4. Finally, we need to work out what $l(x)$ and $g(x)$ - this is where our previous attempts come in, as we can use these to split the hyperparameter values in two groups and then fit some kind of kernel density estimator to each.

# %% papermill={"duration": 12868.370024, "end_time": "2021-12-13T23:25:13.730942", "exception": false, "start_time": "2021-12-13T19:50:45.360918", "status": "completed"} tags=[]
def objective(args,
              train_data):
    
    try:
        print(args)
        int_args = ["num_leaves", "max_depth", "n_estimators"]
        args = {key: int(val) if key in int_args else val for key, val in args.items()}
        args["num_leaves"] = min(args["num_leaves"], 2 ** args["max_depth"])
        model = lightgbm.LGBMClassifier(**args)  
        train_x, train_y = train_data.drop("Cover_Type", axis=1), train_data["Cover_Type"]
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
        es = lightgbm.early_stopping(5, verbose=False)
        model.fit(train_x, train_y, eval_set=(val_x, val_y), callbacks=[es])
        pred = model.predict(val_x)
        return -accuracy_score(pred, val_y)  # -ve so that we max accuracy
    except LightGBMError:
        print("LightGBMError has occurred")
        print(args)
        return 0
        

space = {"max_depth": hyperopt.hp.quniform("max_depth", 3, 8, 1),
         "num_leaves": hyperopt.hp.quniform("num_leaves", 20, 200, 10),
         "learning_rate": hyperopt.hp.loguniform("learning_rate", math.log(0.01), math.log(0.3)),
         "n_estimators": hyperopt.hp.quniform("n_estimators", 50, 200, 10),
         "lambda_l1": hyperopt.hp.uniform("lambda_l1", 0, 100),
         "lambda_l2": hyperopt.hp.uniform("lambda_l2", 0, 100)}
         
fixed_params = {"num_classes": 6,
                "boosting_type": "goss",
                "max_bin": 15, 
                "device": "gpu"}
space.update(fixed_params)

trials = hyperopt.Trials()
best = hyperopt.fmin(lambda x: objective(x, train_data),
                     space=space,
                     algo=hyperopt.tpe.suggest,
                     max_evals=50,
                     trials=trials)


# %% papermill={"duration": 2188.416549, "end_time": "2021-12-14T00:01:42.456894", "exception": false, "start_time": "2021-12-13T23:25:14.040345", "status": "completed"} tags=[]
best_params = {key: int(var) if key in ["num_leaves", "max_depth", "n_estimators", "num_classes"] else var for key, var in best.items()}
fixed_params["max_bin"] = 63
best_params.update(fixed_params)
models, _ = get_trained_models(train_data, partial(get_lightgbm_model, params=best_params))
sub = create_submission(test_data, models)
sub.to_csv("submission.csv", index=False)
