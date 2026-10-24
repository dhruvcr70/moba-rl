from moba_rl.simulation.game import Game
from moba_rl.simulation.lini_tities import Position, Team


def test_minions_can_destroy_enemy_base():
    game = Game(seed=42)

    # Remove the RED tower so BLUE minions can attack the RED base.
    game.towers[Team.RED].hp = 0.0

    # Put several BLUE minions directly next to the RED base.
    game._spawn_minion_wave(Team.BLUE)

    for minion in game.minions:
        if minion.team == Team.BLUE:
            minion.position = Position(
                game.bases[Team.RED].position.x,
                game.bases[Team.RED].position.y,
            )

    # Speed up their attacks for the deterministic test.
    for minion in game.minions:
        if minion.team == Team.BLUE:
            minion.attack_timer = 0.0
            minion.attack_cooldown = 0.0

    # Run until the base is destroyed.
    for _ in range(200):
        game.step(0.1)

        if game.game_over:
            break

    assert game.game_over is True
    assert game.winner == Team.BLUE
    assert game.bases[Team.RED].hp <= 0.0


def test_base_cannot_be_attacked_before_tower_is_destroyed():
    game = Game(seed=42)

    blue = game.champions[Team.BLUE]

    # Move BLUE champion next to RED base.
    red_base = game.bases[Team.RED]

    blue.position = Position(
        red_base.position.x,
        red_base.position.y,
    )

    # RED tower is still alive.
    assert game.towers[Team.RED].alive

    attacked = game.attack_base(
        Team.BLUE,
        Team.RED,
    )

    assert attacked is False
    assert game.bases[Team.RED].hp == 5000.0
    assert game.game_over is False


def test_champion_can_destroy_enemy_base_after_tower_falls():
    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red_base = game.bases[Team.RED]

    # Destroy RED tower.
    game.towers[Team.RED].hp = 0.0

    # Put BLUE champion beside RED base.
    blue.position = Position(
        red_base.position.x,
        red_base.position.y,
    )

    # Make attacks happen immediately.
    blue.attack_timer = 0.0
    blue.attack_cooldown = 0.0

    # Repeatedly attack until the base falls.
    for _ in range(200):
        game.attack_base(
            Team.BLUE,
            Team.RED,
        )

        if game.game_over:
            break

    assert game.game_over is True
    assert game.winner == Team.BLUE
    assert red_base.hp <= 0.0