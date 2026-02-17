from gymnasium.envs.registration import register

from xmorl.envs.sdst.stochastic_deep_sea_treasure import (
    CONCAVE_MAP,
    MIRRORED_MAP,
)


register(
    id="stochastic-deep-sea-treasure-v0",
    entry_point="xmorl.envs.sdst.stochastic_deep_sea_treasure:StochasticDeepSeaTreasure",
    max_episode_steps=100,
)

register(
    id="stochastic-deep-sea-treasure-concave-v0",
    entry_point="xmorl.envs.sdst.stochastic_deep_sea_treasure:StochasticDeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": CONCAVE_MAP},
)

register(
    id="stochastic-deep-sea-treasure-mirrored-v0",
    entry_point="xmorl.envs.sdst.stochastic_deep_sea_treasure:StochasticDeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": MIRRORED_MAP},
)