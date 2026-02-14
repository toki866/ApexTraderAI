# -*- coding: utf-8 -*-
import argparse
import glob
import os
import time
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='SOXL')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    pattern = os.path.join(args.output_root, 'stepE', '**', f'stepE_daily_log_*_{args.symbol}.csv')
    paths = sorted(glob.glob(pattern, recursive=True))
    print('found', len(paths))

    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not {'Date','Split','equity'}.issubset(df.columns):
                continue
            t = df[df['Split'].eq('test')].copy()
            if len(t) == 0:
                continue
            t['Date'] = pd.to_datetime(t['Date'])
            t = t.sort_values('Date')
            e0 = float(t['equity'].iloc[0]); e1 = float(t['equity'].iloc[-1])
            m = e1/e0
            dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(p).st_mtime))
            rows.append({
                'path': p,
                'mtime': dt,
                'test_days': int(len(t)),
                'test_start': str(t['Date'].iloc[0].date()),
                'test_end': str(t['Date'].iloc[-1].date()),
                'equity_mult': m,
                'return_pct': (m-1.0)*100.0,
            })
        except Exception:
            pass

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit('no valid test logs found')

    out = out.sort_values(['test_start','test_end','mtime','equity_mult'], ascending=[True,True,False,False])
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 220)
    print(out.to_string(index=False, formatters={
        'equity_mult': lambda x: f'{x:.6f}',
        'return_pct': lambda x: f'{x: .3f}%',
    }))

    out_path = os.path.join(args.output_root, 'stepE', f'stepE_find_old_logs_{args.symbol}.csv')
    out.to_csv(out_path, index=False)
    print('\nwrite', out_path)


if __name__ == '__main__':
    main()
