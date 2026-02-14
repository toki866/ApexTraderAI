# Event tools patch

## Included
- tools/generate_realclose_events.py : patched to append a tail anchor so REALCLOSE events cover the last date.
- tools/dump_event_matches.py : prints matches + unmatched pred/real events for a given window and matching rules.

## Typical workflow
1) Regenerate REALCLOSE events (now includes tail segment):
   python tools\generate_realclose_events.py --output-root output --symbol SOXL --rbw 10 --min-gap 5
2) Compare and identify extra/unmatched events:
   python tools\dump_event_matches.py --pred-events output\stepD\stepD_events_MAMBA_SOXL.csv --real-events output\stepD\stepD_events_REALCLOSE_SOXL.csv --date-from 2022-01-03 --date-to 2022-03-31 --filter-mode overlap --clip-to-window --overlap-metric iou --min-overlap 0.8 --max-mid-diff-days 10 --require-same-dir
