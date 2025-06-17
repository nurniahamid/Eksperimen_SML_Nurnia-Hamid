
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Fitur tambahan: menghitung umur mobil
    df['Car_Age'] = 2020 - df['Year']
    df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

    # Pisahkan fitur dan target
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    # Daftar fitur numerik dan kategorikal
    numerical_cols = ['Present_Price', 'Kms_Driven', 'Car_Age']
    categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

    # ColumnTransformer untuk preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])

    # Transformasi data
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y
