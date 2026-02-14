# ai_core/rl/algo_marl.py

"""
Deprecated shim module.

今後は必ず:

    from ai_core.rl.algos import MARLAlgo

を使ってください。

このモジュールは、既存コードにある
    from ai_core.rl.algo_marl import MARLAlgo
という import を壊さないための互換レイヤーです。
"""

from ai_core.rl.algos import MARLAlgo

__all__ = ["MARLAlgo"]
