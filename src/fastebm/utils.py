import ruptures as rpt
import numpy as np

from ._changepoint import l2_one_breakpoint_indices
def changepoint_detect_and_order(df, anamoly_model='l2'):
    event_order_dict = {}
    if anamoly_model == 'l2':
        if df.shape[1] == 0:
            return []

        values = df.to_numpy(dtype=float)
        if not np.isnan(values).any():
            breakpoints = l2_one_breakpoint_indices(values, min_size=2)
            for col, breakpoint in zip(df.columns, breakpoints):
                if breakpoint < 0:
                    event_order_dict[col] = float("inf")
                    continue
                breakpoint = min(max(int(breakpoint), 0), len(df) - 1)
                event_order_dict[col] = df.index[breakpoint]

            return sorted(event_order_dict, key=event_order_dict.get)

        for col in df.columns:
            s = df[col].dropna()

            if len(s) < 2:
                event_order_dict[col] = float("inf")
                continue

            values = s.to_numpy(dtype=float)
            breakpoint = l2_one_breakpoint_indices(values, min_size=2)[0]
            if breakpoint < 0:
                event_order_dict[col] = float("inf")
                continue

            breakpoint = min(max(int(breakpoint), 0), len(s) - 1)
            event_order_dict[col] = s.index[breakpoint]

        return sorted(event_order_dict, key=event_order_dict.get)

    anamoly_alg = rpt.Dynp(model=anamoly_model)

    for col in df.columns:
        s = df[col].dropna()

        if len(s) < 2:
            event_order_dict[col] = float("inf")
            continue

        values = s.to_numpy(copy=False)
        anamoly_alg.fit(values)
        cp = anamoly_alg.predict(n_bkps=1)[0]

        # Map the breakpoint back to the original index scale
        cp = min(cp, len(s) - 1)
        event_order_dict[col] = s.index[cp]

    event_ordering = sorted(event_order_dict, key=event_order_dict.get)
    return event_ordering
