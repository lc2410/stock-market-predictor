"""Machine learning model utilities for forecasting pipelines: technical indicator calculation and sklearn model wrappers."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

def calculate_log_returns(data, col="Close", forward=False, periods=1):
    """Calculates logarithmic returns for a given column (backward or forward looking)."""
    if forward:
        return np.log(data[col].shift(-periods) / data[col])
    return np.log(data[col] / data[col].shift(periods))

def extract_quantiles_metrics(clf, reg_median, reg_lower, reg_upper, test_row, predictors, today_val):
    """
    Extracts direction, confidence, and bounds from the models.
    Regressor defines the amount and direction; classifier defines direction confidence.
    """
    mean_log_return = float(reg_median.predict(test_row[predictors])[0])
    lower_log_return = float(reg_lower.predict(test_row[predictors])[0])
    upper_log_return = float(reg_upper.predict(test_row[predictors])[0])
    
    # Natively align direction with regressor outcome
    direction = 1 if mean_log_return > 0 else 0
    
    # Extract classifier confidence
    proba = clf.predict_proba(test_row[predictors])[0]
    if len(clf.classes_) == 2 and clf.classes_[1] == 1:
        prob_up = float(proba[1])
    else:
        prob_up = 1.0 if clf.classes_[0] == 1 else 0.0
        
    dir_conf_final = prob_up if direction == 1 else (1.0 - prob_up)
    
    # Process Bounds
    margin = abs((upper_log_return - lower_log_return) / 2.0)
    final_upper_log = mean_log_return + margin
    final_lower_log = mean_log_return - margin
        
    forecasted_amount = float(today_val * np.exp(mean_log_return))
    amount_lower = float(today_val * np.exp(final_lower_log))
    amount_upper = float(today_val * np.exp(final_upper_log))
    
    # Enforce minimum visual margin (max of 2% or $0.01) to prevent flat bounds in UI
    min_margin = max(forecasted_amount * 0.02, 0.01)
    
    if (amount_upper - forecasted_amount) < min_margin:
        amount_upper = forecasted_amount + min_margin
        amount_lower = max(0.0, forecasted_amount - min_margin) # Prevent negative values
    
    return {
        "Direction": "Up" if direction == 1 else "Down",
        "Direction_Confidence": round(dir_conf_final * 100, 2),
        "Amount": round(forecasted_amount, 2),
        "Amount_Lower": round(amount_lower, 2),
        "Amount_Upper": round(amount_upper, 2)
    }

def fit_models(clf_base, reg_median, reg_lower, reg_upper, valid_all, predictors, col_class, col_reg):
    """Helper to fit classifiers and quantile regressors."""
    class_counts = valid_all[col_class].value_counts()
    min_class_count = class_counts.min() if len(class_counts) > 1 else 0
    
    if min_class_count >= 2:
        cv_folds = min(3, min_class_count)
        clf = CalibratedClassifierCV(estimator=clf_base, method='isotonic', cv=cv_folds)
    else:
        clf = clf_base
        
    clf.fit(valid_all[predictors], valid_all[col_class])
    reg_median.fit(valid_all[predictors], valid_all[col_reg])
    reg_lower.fit(valid_all[predictors], valid_all[col_reg])
    reg_upper.fit(valid_all[predictors], valid_all[col_reg])
    return clf

def init_models(learning_rate, max_depth, min_samples_leaf, max_iter):
    """Helper to instantiate base classifier and quantile regressors."""
    clf_base = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_iter=max_iter,
        class_weight="balanced", random_state=1
    )
    reg_median = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.5, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_iter=max_iter, random_state=1
    )
    reg_lower = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.1, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_iter=max_iter, random_state=1
    )
    reg_upper = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.9, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_iter=max_iter, random_state=1
    )
    return clf_base, reg_median, reg_lower, reg_upper
