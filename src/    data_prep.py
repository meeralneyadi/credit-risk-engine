import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DATA_RAW, DATA_PROCESSED, ART_MODELS, RAW_XLS_NAME, RANDOM_STATE


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_RAW / RAW_XLS_NAME, header=1)
    df.columns = (
        df.columns.str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )
    df = df.rename(columns={"DEFAULT_PAYMENT_NEXT_MONTH": "DEFAULT"})
    return df


def make_splits_and_scale(
    test_size_total: float = 0.4,  # 0.4 => val+test (20/20)
):
    df = load_raw()
    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]

    # 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size_total,
        random_state=RANDOM_STATE,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(ART_MODELS, exist_ok=True)

    X_train_scaled.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_val_scaled.to_csv(DATA_PROCESSED / "X_val.csv", index=False)
    X_test_scaled.to_csv(DATA_PROCESSED / "X_test.csv", index=False)

    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_val.to_csv(DATA_PROCESSED / "y_val.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    joblib.dump(scaler, ART_MODELS / "scaler.joblib")

    return {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "default_rate": float(y.mean())
    }


if __name__ == "__main__":
    info = make_splits_and_scale()
    print("Saved processed splits + scaler:", info)
