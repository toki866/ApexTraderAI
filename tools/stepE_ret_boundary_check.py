# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='SOXL')
    ap.add_argument('--mode', default='sim')
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    pattern = os.path.join(args.output_root, 'stepE', args.mode, f'stepE_daily_log_*_{args.symbol}.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'No files matched: {pattern}')

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        need = {'Date', 'Split', 'equity', 'ret'}
        if not need.issubset(df.columns):
            continue
        t = df[df['Split'].eq('test')].copy()
        if len(t) < 2:
            continue
        t['Date'] = pd.to_datetime(t['Date'])
        t = t.sort_values('Date')

        r = pd.to_numeric(t['ret'], errors='coerce').fillna(0.0)
        e0 = float(t['equity'].iloc[0])
        e1 = float(t['equity'].iloc[-1])

        agent = os.path.basename(p).replace('stepE_daily_log_', '').replace(f'_{args.symbol}.csv', '')
        rows.append({
            'agent': agent,
            'test_days': int(len(t)),
            'test_start': str(t['Date'].iloc[0].date()),
            'test_end': str(t['Date'].iloc[-1].date()),
            'equity_mult': e1 / e0,
            'prod_all': float((1.0 + r).prod()),
            'prod_skip1': float((1.0 + r.iloc[1:]).prod()),
            'diff_all_minus_equity': float((1.0 + r).prod()) - (e1 / e0),
            'diff_skip1_minus_equity': float((1.0 + r.iloc[1:]).prod()) - (e1 / e0),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values('equity_mult', ascending=False)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 200)
    print(out.to_string(index=False, formatters={
        'equity_mult': lambda x: f'{x:.6f}',
        'prod_all': lambda x: f'{x:.6f}',
        'prod_skip1': lambda x: f'{x:.6f}',
        'diff_all_minus_equity': lambda x: f'{x:+.6f}',
        'diff_skip1_minus_equity': lambda x: f'{x:+.6f}',
    }))

    out_path = os.path.join(args.output_root, 'stepE', args.mode, f'stepE_ret_boundary_check_{args.symbol}.csv')
    out.to_csv(out_path, index=False)
    print('\nwrite', out_path)


if __name__ == '__main__':
    main()
