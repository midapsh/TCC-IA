import pandas as pd

from utils.load_df import load_df


def get_bauru_df() -> pd.DataFrame:
    df = load_df([65554853])  # BAURU
    return df


def main():
    df = get_bauru_df()


if __name__ == "__main__":
    main()
