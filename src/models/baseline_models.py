"""
===========================================================
Baseline Models Module

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. MLP Neural Network

Purpose:
Create baseline ML models for comparison
with proposed XGBoost model.

===========================================================
"""


# ==========================================================
# IMPORTS
# ==========================================================


from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingModuleSource]

from sklearn.tree import DecisionTreeClassifier  # pyright: ignore[reportMissingModuleSource]

from sklearn.ensemble import RandomForestClassifier  # pyright: ignore[reportMissingModuleSource]

from sklearn.neural_network import MLPClassifier  # pyright: ignore[reportMissingModuleSource]


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

def create_logistic_model():

    model = LogisticRegression(

        max_iter=1000,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1

    )

    return model


# ==========================================================
# DECISION TREE
# ==========================================================

def create_decision_tree_model():

    model = DecisionTreeClassifier(

        random_state=42,

        class_weight="balanced",

        max_depth=20

    )

    return model


# ==========================================================
# RANDOM FOREST
# ==========================================================

def create_random_forest_model():

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=20

    )

    return model


# ==========================================================
# MLP NEURAL NETWORK
# ==========================================================

def create_mlp_model():

    model = MLPClassifier(

        hidden_layer_sizes=(128, 64),

        activation="relu",

        solver="adam",

        batch_size=512,

        learning_rate="adaptive",

        max_iter=50,

        random_state=42

    )

    return model


# ==========================================================
# MODEL COLLECTION
# ==========================================================

def get_all_baseline_models():

    models = {


        "Logistic Regression":

            create_logistic_model(),



        "Decision Tree":

            create_decision_tree_model(),



        "Random Forest":

            create_random_forest_model(),



        "MLP":

            create_mlp_model()

    }

    return models
