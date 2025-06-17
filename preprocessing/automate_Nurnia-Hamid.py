import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)

    # Feature engineering
    df['Car_Age'] = 2020 - df['Year']
    df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

    # Features and target
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    # Columns by type
    numerical_cols = ['Present_Price', 'Kms_Driven', 'Car_Age']
    categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])

    # Fit and transform
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y

def main(data_path):
    X, y = load_and_preprocess_data(data_path)
    print(f"✅ Data berhasil diproses: {X.shape[0]} baris dan {X.shape[1]} fitur.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path ke file CSV")
    args = parser.parse_args()
    main(args.data_path)

