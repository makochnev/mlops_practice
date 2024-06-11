import pandas as pd

# Import libs to solve classification task
from catboost import CatBoostClassifier


# Make prediction
def make_pred(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = CatBoostClassifier()
    model.load_model('./models/model')

    # Define optimal threshold
    threshold = 0.35

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (model.predict_proba(dt)[:, 1] > threshold) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return submission