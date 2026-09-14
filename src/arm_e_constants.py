"""Arm E's architecture and the two storage figures of pre-registration section 6.

Fixed before any training and never revisited. Regenerate with
`scripts/select_arm_e.py --write`; `--check` asserts the code still agrees with
these committed values.
"""

ARM_E_DEPTH = 44
ARM_E_N = 7
ARM_E_PARAMETERS = 658586
ARM_E_STORAGE_BITS = 21176128
TERNARY_R18_STORAGE_BITS = 23300416
SELECTION_TABLE = [
    {
        "depth": 20,
        "n": 3,
        "parameters": 269722,
        "storage_bits": 8675136,
        "storage_mb": 1.0341567993164062,
        "rel_error": 0.6276832138962669
    },
    {
        "depth": 32,
        "n": 5,
        "parameters": 464154,
        "storage_bits": 14925632,
        "storage_mb": 1.7792739868164062,
        "rel_error": 0.35942637247334985
    },
    {
        "depth": 44,
        "n": 7,
        "parameters": 658586,
        "storage_bits": 21176128,
        "storage_mb": 2.5243911743164062,
        "rel_error": 0.09116953105043275
    },
    {
        "depth": 56,
        "n": 9,
        "parameters": 853018,
        "storage_bits": 27426624,
        "storage_mb": 3.2695083618164062,
        "rel_error": 0.17708731037248435
    },
    {
        "depth": 110,
        "n": 18,
        "parameters": 1727962,
        "storage_bits": 55553856,
        "storage_mb": 6.622535705566406,
        "rel_error": 1.3842430967756112
    }
]
