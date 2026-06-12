import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.base import clone

def rolling_sgd_predictions(df, features, target='next_open_return',
                            window=10, alpha=0.01, retrain_interval=3):
    
    df = df.copy()
    df['combined_signal'] = np.nan
    
    # Create base model outside loop
    base_model = SGDRegressor(
        alpha=alpha,
        penalty='l2',
        max_iter=1,
        tol=None,
        warm_start=True,
        learning_rate='constant',
        eta0=0.01,
        random_state=42,
    )
    
    for sym, sym_df in df.groupby('symbol', sort=False):
        X = sym_df[features].values
        y = sym_df[target].values
        
        model = clone(base_model)
        preds = []
        
        i = window
        while i < len(sym_df):
            model.partial_fit(X[i - window:i], y[i - window:i])
            
            batch_end = min(i + retrain_interval, len(sym_df))
            
            batch_indices = range(i, batch_end)  
            batch_X = X[batch_indices]
            
            batch_predictions = model.predict(batch_X)
            
            for pred in batch_predictions:
                preds.append(pred)
            
            i = batch_end
        
        df.loc[sym_df.index[window:], 'combined_signal'] = preds
    
    return df
