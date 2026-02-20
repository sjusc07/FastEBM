import numpy as np
import ruptures as rpt

def changepoint_detect_and_order(df, anamoly_model = 'l2'):
    event_order_dict = {}
    feature_names = list(df.columns)
    anamoly_alg = rpt.Dynp(model = anamoly_model)
    for col in feature_names:
        anamoly_alg.fit(df[col].dropna().values)
        event_order_dict[col] = anamoly_alg.predict(n_bkps = 1)[0]
    event_ordering = list({k: v for k, v in sorted(event_order_dict.items(), key=lambda item: item[1])}.keys())
    return event_ordering