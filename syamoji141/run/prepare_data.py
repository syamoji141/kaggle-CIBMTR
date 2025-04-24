import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import yaml
from lifelines import KaplanMeierFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.conf import PrepareDataConfig
from src.utils.common import load_from_yaml, trace

RMV = ["ID","efs","efs_time","y"]
CATS = []
CAT_SIZE = []
CAT_EMB = []
NUMS = []

class FE:

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def _load_data(self, path):

        return pl.read_csv(path, batch_size=self._batch_size)

    def _update_hla_columns(self, df):
        
        df = df.with_columns(
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .alias('hla_nmdp_6'),
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_drb1_low').fill_null(0))
            .alias('hla_low_res_6'),
            
            pl.col('hla_match_a_high').fill_null(0)
            .add(pl.col('hla_match_b_high').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .alias('hla_high_res_6'),
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_c_low').fill_null(0))
            .add(pl.col('hla_match_drb1_low').fill_null(0))
            .alias('hla_low_res_8'),
            
            pl.col('hla_match_a_high').fill_null(0)
            .add(pl.col('hla_match_b_high').fill_null(0))
            .add(pl.col('hla_match_c_high').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .alias('hla_high_res_8'),
            
            pl.col('hla_match_a_low').fill_null(0)
            .add(pl.col('hla_match_b_low').fill_null(0))
            .add(pl.col('hla_match_c_low').fill_null(0))
            .add(pl.col('hla_match_drb1_low').fill_null(0))
            .add(pl.col('hla_match_dqb1_low').fill_null(0))
            .alias('hla_low_res_10'),
            
            pl.col('hla_match_a_high').fill_null(0)
            .add(pl.col('hla_match_b_high').fill_null(0))
            .add(pl.col('hla_match_c_high').fill_null(0))
            .add(pl.col('hla_match_drb1_high').fill_null(0))
            .add(pl.col('hla_match_dqb1_high').fill_null(0))
            .alias('hla_high_res_10'),
            
        )
    
        return df

    def _cast_datatypes(self, df):

        self.num_cols = [
            'hla_high_res_8',
            'hla_low_res_8',
            'hla_high_res_6',
            'hla_low_res_6',
            'hla_high_res_10',
            'hla_low_res_10',
            'hla_match_dqb1_high',
            'hla_match_dqb1_low',
            'hla_match_drb1_high',
            'hla_match_drb1_low',
            'hla_nmdp_6',
            'year_hct',
            'hla_match_a_high',
            'hla_match_a_low',
            'hla_match_b_high',
            'hla_match_b_low',
            'hla_match_c_high',
            'hla_match_c_low',
            'donor_age',
            'age_at_hct',
            'comorbidity_score',
            'karnofsky_score',
            'efs',
            'efs_time',
            'y',
        ]

        for col in df.columns:

            if col in self.num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))  

            else:
                df = df.with_columns(pl.col(col).fill_null('Unknown').cast(pl.String))  

        return df.with_columns(pl.col('ID').cast(pl.Int32))


    def apply_fe(self, df):
 
        df = self._update_hla_columns(df)                     
        df = self._cast_datatypes(df)        
        df = df
        
        cat_cols = [col for col in df.columns if (df[col].dtype == pl.String) & (col not in RMV)]
        num_cols = [col for col in self.num_cols if col not in RMV]

        return df, cat_cols, num_cols


def transform_target(df: pl.DataFrame, target_type: str) -> pl.DataFrame:
    if target_type == "chris":
        # --- 1) y = efs_time ---
        df = df.with_columns(
            pl.col("efs_time").alias("y")
        )

        mx = df.filter(pl.col("efs") == 1)["efs_time"].max()
        mn = df.filter(pl.col("efs") == 0)["efs_time"].min()

        df = df.with_columns(
            pl.when(pl.col("efs") == 0)
            .then(pl.col("y") + (mx - mn))
            .otherwise(pl.col("y"))
            .alias("y")
        )

        df = df.with_columns(
            pl.col("y").rank(method="ordinal").alias("y")
        )

        n = df.height 
        df = df.with_columns(
            pl.when(pl.col("efs") == 0)
            .then(pl.col("y") + 2 * n)
            .otherwise(pl.col("y"))
            .alias("y")
        )

        y_max = df["y"].max()
        df = df.with_columns(
            (pl.col("y") / y_max).alias("y")
        )

        df = df.with_columns(
            pl.col("y").log().alias("y")
        )

        y_mean = df["y"].mean()
        df = df.with_columns(
            (pl.col("y") - y_mean).alias("y")
        )

        df = df.with_columns(
            (pl.col("y") * -1).alias("y")
        )
    elif target_type == "kaplan_meier":
        def KaplanMeier(in_data, time_col='efs_time', event_col='efs'):
            kmf = KaplanMeierFitter()
            kmf.fit(durations=in_data[time_col], event_observed=in_data[event_col])
            return kmf.survival_function_at_times(in_data[time_col]).values.flatten()

        def update_target_with_probabilities(df, probability_func, target_name, time_col='efs_time', event_col='efs', sep=0):
            race_group = sorted(df['race_group'].unique())
            probs_dict = {}
            
            # Compute probabilities for each race group
            for race in race_group:
                race_df = df[df['race_group'] == race]
                probs_dict[race] = probability_func(race_df, time_col, event_col)
            
            # Update target values using the target_name parameter
            for race in race_group:
                df.loc[df['race_group'] == race, target_name] = probs_dict[race]
            
            # Adjust target for non-events
            df.loc[df[event_col] == 0, target_name] -= sep
            
            return df[[event_col,target_name]]

        df = df.to_pandas()
        transforms = {}
        transforms["KaplanMeier_015"] = update_target_with_probabilities(df=df, probability_func=KaplanMeier, target_name="KaplanMeier_015", sep=0.15)
        for i, (name, risk) in enumerate(transforms.items()):
            if name == 'KaplanMeier_015':
                print(risk[name])
                df['y'] = risk[name]

        df = pl.from_pandas(df)
        df = df.drop("KaplanMeier_015")
    else:
        raise NotImplementedError(f"target_type: {target_type} is not implemented.")

    return df

def transform_df(
    train: pl.DataFrame,
    test: pl.DataFrame,
    FEATURES: list[str],
    CATS: list[str]
):

    combined = pl.concat([train, test], how="diagonal")
    train_len = train.height

    print("We LABEL ENCODE the CATEGORICAL FEATURES:")
    for c in FEATURES:
        if c in CATS:

            combined = combined.with_columns(
                pl.col(c).cast(pl.Categorical).to_physical().alias(c)
            )

            c_min = combined.select(pl.col(c).min()).item()
            combined = combined.with_columns(
                (pl.col(c) - c_min).alias(c)
            )

            combined = combined.with_columns(
                pl.col(c).cast(pl.Int32)
            )

            n_unique = combined.select(pl.col(c).n_unique()).item()
            c_max = combined.select(pl.col(c).max()).item()

            print(f"{c} has ({n_unique}) unique values")

            CAT_SIZE.append(c_max + 1)
            CAT_EMB.append(int(np.ceil(np.sqrt(c_max + 1))))

        else:

            current_dtype = combined.schema[c]
            if current_dtype == pl.Float64:
                combined = combined.with_columns(
                    pl.col(c).cast(pl.Float32)
                )
            elif current_dtype == pl.Int64:
                combined = combined.with_columns(
                    pl.col(c).cast(pl.Int32)
                )

            c_mean = combined.select(pl.col(c).mean()).item()
            c_std = combined.select(pl.col(c).std()).item()

            combined = combined.with_columns(
                ((pl.col(c) - c_mean) / c_std)
                .fill_null(0)
                .alias(c)
            )

            NUMS.append(c)

    train_new = combined.slice(offset=0, length=train_len)
    test_new = combined.slice(offset=train_len)

    return train_new, test_new

def make_fold(df: pl.DataFrame, n_repeats: int, n_folds: int,  fold_type: str = None)  -> pl.DataFrame:
    if fold_type == "v2":
        df = df.with_columns(
            (pl.col("race_group").cast(pl.String) + "_" + pl.col("efs").cast(pl.String)).alias("race_group_efs")
        )
        train = df.to_pandas()
        for r in range(n_repeats):
            train[f"fold_repeat{r}"] = -1
            skf = StratifiedKFold(n_splits=n_folds, random_state=42+r, shuffle=True)
            for fold, (_, test_index) in enumerate(skf.split(train, train["race_group_efs"])):
                train.loc[test_index, f"fold_repeat{r}"] = fold
        train = pl.from_pandas(train)
    elif fold_type is None:
        train = df.to_pandas()
        for r in range(n_repeats):
            train[f"fold_repeat{r}"] = -1
            skf = StratifiedKFold(n_splits=n_folds, random_state=42+r, shuffle=True)
            for fold, (_, test_index) in enumerate(skf.split(train, train["race_group"])):
                train.loc[test_index, f"fold_repeat{r}"] = fold
        train = pl.from_pandas(train)
    return train

@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.3")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    train = pl.read_csv(Path(cfg.dir.data_dir) / "train.csv")
    test = pl.read_csv(Path(cfg.dir.data_dir) / "test.csv")

    if cfg.phase == "train":
        with trace("transform target"):
            train = transform_target(train, target_type=cfg.target_type)
            print(train)

    with trace("encode categorical feature"):
        if cfg.encoder == "label_encoder":
            features = [col for col in train.columns if not col in RMV]
            print(f"There are {len(features)} FEATURES: {features}")

            for c in features:
                if train[c].dtype==pl.String:
                    train = train.with_columns(pl.col(c).fill_null("NAN").alias(c))
                    test = test.with_columns(pl.col(c).fill_null("NAN").alias(c))
                    CATS.append(c)
                elif not "age" in c:
                    train = train.with_columns(pl.col(c).fill_null("NAN").cast(pl.String).alias(c))
                    test = test.with_columns(pl.col(c).fill_null("NAN").cast(pl.String).alias(c))
                    CATS.append(c)
            print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

            train, test = transform_df(train, test, features, CATS)
        elif cfg.encoder == "v2":
            fe = FE(cfg.batch_size)
            if cfg.phase == "train":
                train, CATS, NUMS = fe.apply_fe(train)
                for c in CATS:
                    train = train.with_columns(
                        pl.col(c).cast(pl.Categorical).to_physical().alias(c)
                    )

                    c_min = train.select(pl.col(c).min()).item()
                    train = train.with_columns(
                        (pl.col(c) - c_min).alias(c)
                    )

                    train = train.with_columns(
                        pl.col(c).cast(pl.Int32)
                    )
                    n_unique = train.select(pl.col(c).n_unique()).item()
                    c_max = train.select(pl.col(c).max()).item()

                    print(f"{c} has ({n_unique}) unique values")

                    CAT_SIZE.append(int(c_max) + 1)
                    CAT_EMB.append(int(np.ceil(np.sqrt(int(c_max) + 1))))
            else:
                test, cats, _ = fe.apply_fe(test)
                for c in cats:
                    test = test.with_columns(
                        pl.col(c).cast(pl.Categorical).to_physical().alias(c)
                    )

                    c_min = test.select(pl.col(c).min()).item()
                    test = test.with_columns(
                        (pl.col(c) - c_min).alias(c)
                    )

                    test = test.with_columns(
                        pl.col(c).cast(pl.Int32)
                    )

        elif cfg.encoder == "v3": # w/ label encoder + scaler(only num cols)
            features = [col for col in train.columns if not col in RMV]
            CATS = []
            CAT_SIZE = []
            CAT_EMB = []
            NUMS = []
            print(f"There are {len(features)} FEATURES: {features}")
            if cfg.phase == "train":
                for c in features:
                    if train[c].dtype==pl.String:
                        train = train.with_columns(pl.col(c).fill_null("NAN").alias(c))
                        CATS.append(c)
                    elif not "age" in c:
                        train = train.with_columns(pl.col(c).fill_null("NAN").cast(pl.String).alias(c))
                        CATS.append(c)
                    else:
                        train = train.with_columns(pl.col(c).fill_null(-1).alias(c))
                        NUMS.append(c)

                cat_values = train.select(CATS).to_numpy()
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                encoded_values = encoder.fit_transform(cat_values)

                train = train.with_columns(
                    [pl.Series(col, encoded_values[:, i] + 1).cast(pl.Int32) for i, col in enumerate(CATS)]
                )

                for c in CATS:
                    c_max = train.select(pl.col(c).max()).item()
                    CAT_SIZE.append(c_max + 1)
                    CAT_EMB.append(int(np.ceil(np.sqrt(c_max + 1))))

                # make fold
                train = make_fold(train, cfg.repeats, cfg.folds, cfg.fold_type)

                scaler = StandardScaler()
                values = train.select(NUMS).to_numpy()
                scaled_values = scaler.fit_transform(values)
                train = train.with_columns(
                    [pl.Series(col, scaled_values[:, i]) for i, col in enumerate(NUMS)]
                )

                with open(Path(cfg.dir.model_dir) / "v3_ordinal_encoder.pkl", "wb") as f:
                    pickle.dump(encoder, f)

                with open(Path(cfg.dir.model_dir) / "v3_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)

            else:
                for c in features:
                    if test[c].dtype==pl.String:
                        test = test.with_columns(pl.col(c).fill_null("NAN").alias(c))
                    elif not "age" in c:
                        test = test.with_columns(pl.col(c).fill_null("NAN").cast(pl.String).alias(c))
                    else:
                        test = test.with_columns(pl.col(c).fill_null(-1).alias(c))

                data = load_from_yaml(Path(cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml")
                CATS = data["CATS"]
                NUMS = data["NUMS"]

                with open(Path(cfg.dir.model_dir) / "v3_ordinal_encoder.pkl", "rb") as f:
                    encoder = pickle.load(f)

                cat_values = test.select(CATS).to_numpy()
                encoded_values = encoder.transform(cat_values)

                test = test.with_columns(
                    [pl.Series(col, encoded_values[:, i] + 1).cast(pl.Int32) for i, col in enumerate(CATS)]
                )

                with open(Path(cfg.dir.model_dir) / "v3_scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                
                values = test.select(NUMS).to_numpy()
                scaled_values = scaler.transform(values)
                test = test.with_columns(
                    [pl.Series(col, scaled_values[:, i]) for i, col in enumerate(NUMS)]
                )


            print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
        elif cfg.encoder == "v4": # w/ scaler + label encoder
            features = [col for col in train.columns if not col in RMV]
            CATS = []
            CAT_SIZE = []
            CAT_EMB = []
            NUMS = []
            print(f"There are {len(features)} FEATURES: {features}")
            if cfg.phase == "train":
                for c in features:
                    if train[c].dtype==pl.String:
                        train = train.with_columns(pl.col(c).fill_null("NAN").alias(c))
                        CATS.append(c)
                    elif not "age" in c:
                        train = train.with_columns(pl.col(c).fill_null("NAN").cast(pl.String).alias(c))
                        CATS.append(c)
                    else:
                        train = train.with_columns(pl.col(c).fill_null(-1).alias(c))
                        NUMS.append(c)
                        

                cat_values = train.select(CATS).to_numpy()
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                encoded_values = encoder.fit_transform(cat_values)

                train = train.with_columns(
                    [pl.Series(col, encoded_values[:, i] + 1).cast(pl.Int32) for i, col in enumerate(CATS)]
                )

                for c in CATS:
                    c_max = train.select(pl.col(c).max()).item()
                    # カテゴリ数と埋め込み次元のリストを更新
                    CAT_SIZE.append(c_max + 1)
                    CAT_EMB.append(int(np.ceil(np.sqrt(c_max + 1))))

                # make fold
                train = train.to_pandas()
                for r in range(cfg.repeats):
                    train[f"fold_repeat{r}"] = -1
                    skf = StratifiedKFold(n_splits=cfg.folds, random_state=42+r, shuffle=True)
                    for fold, (_, test_index) in enumerate(skf.split(train, train["race_group"])):
                        train.loc[test_index, f"fold_repeat{r}"] = fold
                train = pl.from_pandas(train)

                scaler = StandardScaler()
                values = train.select(features).to_numpy()
                scaled_values = scaler.fit_transform(values)
                train = train.with_columns(
                    [pl.Series(col, scaled_values[:, i]) for i, col in enumerate(features)]
                )

                with open(Path(cfg.dir.model_dir) / "ordinal_encoder.pkl", "wb") as f:
                    pickle.dump(encoder, f)

                with open(Path(cfg.dir.model_dir) / "scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)

            else:
                data = load_from_yaml(Path(cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml")
                CATS = data["CATS"]

                with open(Path(cfg.dir.model_dir) / "ordinal_encoder.pkl", "rb") as f:
                    encoder = pickle.load(f)

                cat_values = test.select(CATS).to_numpy()
                encoded_values = encoder.transform(cat_values)

                test = test.with_columns(
                    [pl.Series(col, encoded_values[:, i] + 1).cast(pl.Int32) for i, col in enumerate(CATS)]
                )

                with open(Path(cfg.dir.model_dir) / "scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                
                values = test.select(features).to_numpy()
                scaled_values = scaler.transform(values)
                test = test.with_columns(
                    [pl.Series(col, scaled_values[:, i]) for i, col in enumerate(features)]
                )


            print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")
        else:
            raise NotImplementedError(f"{cfg.encoder} is not implemented in cfg.encoder")

    if cfg.phase == "train":
        train.write_csv(Path(cfg.dir.processed_dir) / "train.csv")
    else:
        test.write_csv(Path(cfg.dir.processed_dir) / "test.csv")

    if cfg.phase == "train":
        data = {
            "RMV": RMV,
            "CATS": CATS,
            "CAT_SIZE": CAT_SIZE,
            "CAT_EMB": CAT_EMB,
            "NUMS": NUMS,
        }
        
        with open(Path(cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        print(train)
    else:
        print(test)

if __name__ == "__main__":
    main()