import ruptures as rpt

def _ordered_event_dict(df, anamoly_model='l2', feature_names=None):
    event_order_dict = {}
    if feature_names is None:
        feature_names = list(df.columns)

    for col in feature_names:
        series = df[col]
        valid_mask = series.notna().to_numpy()
        valid_positions = valid_mask.nonzero()[0]
        values = series.to_numpy()[valid_mask]

        if len(values) < 2:
            raise ValueError(
                f"Feature '{col}' has fewer than 2 non-missing values after sorting."
            )

        anamoly_alg = rpt.Dynp(model=anamoly_model)
        anamoly_alg.fit(values)

        try:
            reduced_breakpoint = anamoly_alg.predict(n_bkps=1)[0]
            event_order_dict[col] = int(valid_positions[reduced_breakpoint])
        except Exception as exc:
            raise ValueError(
                f"Could not detect a changepoint for feature '{col}' with {len(values)} observations."
            ) from exc

    return dict(sorted(event_order_dict.items(), key=lambda item: item[1]))


def changepoint_detect_and_order(df, anamoly_model = 'l2'):
    event_ordering = list(_ordered_event_dict(df, anamoly_model).keys())
    return event_ordering
