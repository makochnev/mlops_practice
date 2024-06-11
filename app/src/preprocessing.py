# Import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Define column types
numerical = ['использование', 'сумма', 'частота_пополнения',
                        'доход', 'сегмент_arpu', 'частота', 'объем_данных',
                        'on_net', 'продукт_1', 'продукт_2', 'секретный_скор',
                        'pack_freq']
categorical = ['регион']
deleted = ['client_id', 'зона_1', 'зона_2', 'mrg_', 'pack']
# target_col = 'binary_target'


def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=deleted)

    return input_df


# Preprocessing function
def preprocess(input_df):

    train = pd.read_csv('./train_data/train.csv')

    input_df['использование'] = input_df['использование'].astype(str).apply(
        lambda x: int(x.split('_')[0]) if x[0].isdigit() else int(x[1:3]))

    for col in numerical:
        input_df[col].fillna(-1, inplace=True)

    for col in categorical:
        input_df[col].fillna('No data', inplace=True)

    onehotencoder = OneHotEncoder(drop='first',
                                  sparse_output=False,
                                  handle_unknown='ignore')
    onehotencoder.fit(train[categorical].fillna('No data'))
    encoded_df = pd.DataFrame(onehotencoder.transform(input_df[categorical]))
    encoded_df.columns = onehotencoder.get_feature_names_out()

    # Create dataframe
    output_df = input_df.drop(columns=categorical, axis=1).join(encoded_df)
 
    # Return resulting dataset
    return output_df
