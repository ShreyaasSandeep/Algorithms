import numpy as np
from sklearn.linear_model import SGDRegressor

def rolling_sgd_predictions(df, features, target='next_open_return',
                            window=252, alpha=0.01, retrain_interval=5):

    df = df.copy()
    df['combined_signal'] = np.nan

    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym].copy()
        X = sym_df[features].values
        y = sym_df[target].values
        preds = []

        model = SGDRegressor(
            alpha=alpha,
            penalty='l2',
            max_iter=1,
            tol=None,
            warm_start=True,
            learning_rate='constant',
            eta0=0.01,
            random_state=42,
        )

        for i in range(window, len(sym_df)):
            if (i - window) % retrain_interval == 0:
                model.partial_fit(X[i - window:i], y[i - window:i])
            preds.append(model.predict(X[[i]])[0])

        df.loc[sym_df.index[window:], 'combined_signal'] = preds

    return df
