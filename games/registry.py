from __future__ import annotations

from games.a.game_a import GAME_INFO as GAME_A_INFO
from games.b.game_b import GAME_INFO as GAME_B_INFO
from games.game_controller import GameInfo


GAMES: dict[str, GameInfo] = {info.game_id: info for info in (GAME_A_INFO, GAME_B_INFO)}
