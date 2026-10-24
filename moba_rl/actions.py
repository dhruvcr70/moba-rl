ACTION_NAMES = {
    0: "wait",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "attack_champion",
    6: "attack_tower",
    7: "ability_1",
    8: "ability_2",
    9: "ability_3",
    10: "attack_objective",
    11: "attack_base",
    12: "farm_minion",
    13: "buy_blade",
    14: "buy_armor",
    15: "buy_boots",
    16: "buy_vitality",
    17: "engage_champion",
    18: "push_lane",
    19: "contest_objective",
    20: "farm_lane",
    21: "return_to_shop",
    22: "defend_lane",
    23: "retreat_to_base",
    24: "rotate_top",
    25: "rotate_mid",
    26: "rotate_bottom",
    27: "freeze_wave",
    28: "slow_push",
    29: "fast_push",
    30: "ward_objective",
    31: "sweep_vision",
    32: "zone_objective",
    33: "all_in",
    34: "poke",
    35: "save_cooldowns",
}

ADVANCED_ACTIONS = {
    name: action for action, name in ACTION_NAMES.items() if action >= 24
}

HIERARCHICAL_ACTIONS = {
    "engage_champion": 17,
    "push_lane": 18,
    "contest_objective": 19,
    "farm_lane": 20,
    "return_to_shop": 21,
    "defend_lane": 22,
    "retreat_to_base": 23,
}


def action_name(action):
    try:
        return ACTION_NAMES[int(action)]
    except (KeyError, TypeError, ValueError):
        return f"unknown_{action}"
