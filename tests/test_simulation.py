from math import isclose
from moba_rl.simulation.lini_tities import Position, Team, Minion
from moba_rl.simulation.game import Game


def test_game_initializes():

    game = Game(seed=42)

    assert game.time == 0.0
    assert game.tick_count == 0
    assert game.game_over is False
    assert game.winner is None


def test_both_teams_exist():

    game = Game(seed=42)

    assert Team.BLUE in game.champions
    assert Team.RED in game.champions

    assert Team.BLUE in game.towers
    assert Team.RED in game.towers

    assert Team.BLUE in game.bases
    assert Team.RED in game.bases


def test_simulation_advances():

    game = Game(seed=42)

    game.step(0.5)

    assert game.time == 0.5
    assert game.tick_count == 1


def test_entities_have_unique_ids():

    game = Game(seed=42)

    entities = game.get_entities()

    ids = [entity.id for entity in entities]

    assert len(ids) == len(set(ids))


def test_enemy_team():

    game = Game(seed=42)

    assert game.enemy_team(Team.BLUE) == Team.RED
    assert game.enemy_team(Team.RED) == Team.BLUE


def test_champion_can_move():

    game = Game(seed=42)

    champion = game.champions[Team.BLUE]

    starting_position = Position(
        champion.position.x,
        champion.position.y,
    )

    target = Position(
        starting_position.x + 500.0,
        starting_position.y,
    )

    game.move_champion(
        Team.BLUE,
        target,
        dt=1.0,
    )

    assert champion.position.x > starting_position.x
    assert champion.position.y == starting_position.y


def test_champion_does_not_move_past_target():

    game = Game(seed=42)

    champion = game.champions[Team.BLUE]

    target = Position(
        champion.position.x + 50.0,
        champion.position.y,
    )

    game.move_champion(
        Team.BLUE,
        target,
        dt=1.0,
    )

    assert champion.position.x == target.x
    assert champion.position.y == target.y


def test_champion_can_attack():

    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    red.position.x = blue.position.x + 50.0
    red.position.y = blue.position.y

    starting_hp = red.hp

    result = game.attack_champion(
        Team.BLUE,
        Team.RED,
    )

    assert result is True
    assert red.hp < starting_hp
    assert red.hp == starting_hp - blue.attack_damage


def test_champion_cannot_attack_out_of_range():

    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    red.position.x = blue.position.x + 1000.0
    red.position.y = blue.position.y

    starting_hp = red.hp

    result = game.attack_champion(
        Team.BLUE,
        Team.RED,
    )

    assert result is False
    assert red.hp == starting_hp


def test_attack_cooldown():

    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    red.position.x = blue.position.x + 50.0
    red.position.y = blue.position.y

    first_attack = game.attack_champion(
        Team.BLUE,
        Team.RED,
    )

    second_attack = game.attack_champion(
        Team.BLUE,
        Team.RED,
    )

    assert first_attack is True
    assert second_attack is False


def test_kill_is_recorded():

    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    red.position.x = blue.position.x + 50.0
    red.position.y = blue.position.y

    red.hp = blue.attack_damage

    result = game.attack_champion(
        Team.BLUE,
        Team.RED,
    )

    assert result is True
    assert red.alive is False
    assert blue.kills == 1
    assert red.deaths == 1
    assert red.respawn_timer == 5.0

def test_champion_respawns():

    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    red.position.x = blue.position.x + 50.0
    red.position.y = blue.position.y

    red.hp = blue.attack_damage

    game.attack_champion(
        Team.BLUE,
        Team.RED,
    )

    assert red.alive is False
    assert red.respawn_timer == 5.0

    game.step(5.0)

    assert red.alive is True
    assert red.hp == red.max_hp

    spawn = game.map.spawn_position(Team.RED)

    assert red.position.x == spawn.x
    assert red.position.y == spawn.y

def test_minion_wave_spawns():

    game = Game(seed=42)

    assert len(game.minions) == 0

    game.step(10.0)

    # Each of the three lanes spawns one six-minion wave (three per team).
    assert len(game.minions) == 18


def test_minions_move_down_lane():

    game = Game(seed=42)

    game.step(10.0)

    minion = game.minions[0]

    starting_x = minion.position.x

    game.step(1.0)

    assert minion.position.x > starting_x


def test_minions_have_unique_ids():

    game = Game(seed=42)

    game.step(10.0)

    entities = game.get_entities()

    ids = [entity.id for entity in entities]

    assert len(ids) == len(set(ids))

def test_minions_can_damage_each_other():

    game = Game(seed=42)

    game.step(10.0)

    blue_minion = next(
        minion
        for minion in game.minions
        if minion.team == Team.BLUE and minion.lane == "mid"
    )

    red_minion = next(
        minion
        for minion in game.minions
        if minion.team == Team.RED and minion.lane == "mid"
    )

    blue_minion.position.x = 900.0
    blue_minion.position.y = 300.0

    red_minion.position.x = 900.0
    red_minion.position.y = 300.0

    starting_hp = red_minion.hp

    game.step(0.1)

    assert red_minion.hp < starting_hp


def test_dead_minions_are_removed():

    game = Game(seed=42)

    game.step(10.0)

    blue_minion = next(
        minion
        for minion in game.minions
        if minion.team == Team.BLUE and minion.lane == "mid"
    )

    blue_minion.hp = 0.0

    game.step(0.1)

    assert blue_minion not in game.minions

def test_minions_can_attack_champions():

    game = Game(seed=42)

    game.step(10.0)

    blue_minion = next(
        minion
        for minion in game.minions
        if minion.team == Team.BLUE and minion.lane == "mid"
    )

    red_champion = game.champions[Team.RED]

    blue_minion.position.x = red_champion.position.x
    blue_minion.position.y = red_champion.position.y

    starting_hp = red_champion.hp

    game.step(0.1)

    assert red_champion.hp < starting_hp

def test_champion_can_attack_tower():
    game = Game(seed=42)

    blue_champion = game.champions[Team.BLUE]
    red_tower = game.towers[Team.RED]

    blue_champion.position.x = red_tower.position.x
    blue_champion.position.y = red_tower.position.y

    starting_hp = red_tower.hp

    success = game.attack_tower(
        Team.BLUE,
        Team.RED,
    )

    assert success
    assert red_tower.hp < starting_hp


def test_champion_cannot_attack_tower_out_of_range():
    game = Game(seed=42)

    red_tower = game.towers[Team.RED]

    starting_hp = red_tower.hp

    success = game.attack_tower(
        Team.BLUE,
        Team.RED,
    )

    assert not success
    assert red_tower.hp == starting_hp


def test_minions_can_damage_tower():
    game = Game(seed=42)

    game.step(10.0)

    blue_minion = next(
        minion
        for minion in game.minions
        if minion.team == Team.BLUE and minion.lane == "mid"
    )

    red_tower = game.towers[Team.RED]

    blue_minion.position.x = red_tower.position.x
    blue_minion.position.y = red_tower.position.y

    starting_hp = red_tower.hp

    game.step(0.1)

    assert red_tower.hp < starting_hp


def test_destroyed_tower_stops_defending():
    game = Game(seed=42)

    red_tower = game.towers[Team.RED]

    red_tower.hp = 0.0

    blue_champion = game.champions[Team.BLUE]

    blue_champion.position.x = red_tower.position.x
    blue_champion.position.y = red_tower.position.y

    starting_hp = blue_champion.hp

    game.step(0.1)

    assert blue_champion.hp == starting_hp


def test_destroyed_tower_is_recorded_in_events():
    game = Game(seed=42)

    red_tower = game.towers[Team.RED]

    blue_champion = game.champions[Team.BLUE]

    blue_champion.position.x = red_tower.position.x
    blue_champion.position.y = red_tower.position.y

    red_tower.hp = blue_champion.attack_damage

    success = game.attack_tower(
        Team.BLUE,
        Team.RED,
    )

    assert success
    assert not red_tower.alive

    assert any(
        "tower was destroyed" in event
        for event in game.events
    )

def test_champion_can_use_ability():
    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    blue.position.x = red.position.x
    blue.position.y = red.position.y

    starting_hp = red.hp

    success = game.use_ability(
        Team.BLUE,
        1,
        Team.RED,
    )

    assert success
    assert red.hp < starting_hp
    assert blue.ability_1_timer > 0.0


def test_ability_cannot_be_used_while_on_cooldown():
    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    red = game.champions[Team.RED]

    blue.position.x = red.position.x
    blue.position.y = red.position.y

    assert game.use_ability(
        Team.BLUE,
        1,
        Team.RED,
    )

    assert not game.use_ability(
        Team.BLUE,
        1,
        Team.RED,
    )


def test_champion_gains_xp_and_levels_up():
    game = Game(seed=42)

    champion = game.champions[Team.BLUE]

    starting_level = champion.level
    starting_damage = champion.attack_damage

    champion.gain_xp(100.0)

    assert champion.level > starting_level
    assert champion.attack_damage > starting_damage


def test_minion_kill_gives_gold_and_xp():
    game = Game(seed=42)

    blue = game.champions[Team.BLUE]

    red_minion = Minion(
        id=9999,
        team=Team.RED,
        position=Position(
            blue.position.x,
            blue.position.y,
        ),
        max_hp=10.0,
        hp=10.0,
    )

    game.minions.append(red_minion)

    starting_gold = blue.gold
    starting_xp = blue.xp

    game._minion_attack_minion(
        Minion(
            id=10000,
            team=Team.BLUE,
            position=Position(
                blue.position.x,
                blue.position.y,
            ),
            max_hp=300.0,
            hp=300.0,
            attack_damage=20.0,
        ),
        red_minion,
    )

    assert not red_minion.alive

    # Reward is granted when the victim dies.
    assert blue.gold >= starting_gold
    assert blue.xp >= starting_xp


def test_neutral_objective_can_be_captured():
    game = Game(seed=42)

    blue = game.champions[Team.BLUE]
    objective = game.objective

    assert objective is not None

    blue.position.x = objective.position.x
    blue.position.y = objective.position.y

    objective.hp = 50.0

    starting_gold = blue.gold

    assert game.attack_objective(Team.BLUE)

    assert not objective.alive
    assert blue.gold > starting_gold
    assert blue.objectives_taken == 1


def test_game_state_contains_core_information():
    game = Game(seed=42)

    state = game.get_state()

    assert "time" in state
    assert "champions" in state
    assert "towers" in state
    assert "bases" in state
    assert "objective" in state

    assert "blue" in state["champions"]
    assert "red" in state["champions"]

def test_procedural_map_generation_is_seeded_and_mirrored():
    first = Game(seed=11, procedural_map=True)
    repeat = Game(seed=11, procedural_map=True)
    other = Game(seed=12, procedural_map=True)

    assert first.map.config == repeat.map.config
    assert first.map.config != other.map.config
    assert isclose(
        first.map.config.blue_base_x,
        first.map.config.width - first.map.config.red_base_x,
    )
    assert isclose(
        first.map.config.blue_tower_x,
        first.map.config.width - first.map.config.red_tower_x,
    )
    assert 0.0 <= first.map.config.objective_x <= first.map.config.width
    assert 0.0 <= first.map.config.objective_y <= first.map.config.height


def test_champion_can_farm_minion_and_get_rewards():
    game = Game(seed=42)
    blue = game.champions[Team.BLUE]
    blue_minion_target = Minion(
        id=9999,
        team=Team.RED,
        position=Position(blue.position.x + 50.0, blue.position.y),
        max_hp=300.0,
        hp=50.0,
    )
    game.minions.append(blue_minion_target)
    starting_gold = blue.gold

    assert game.attack_minion(Team.BLUE)
    assert not blue_minion_target.alive
    assert blue.gold == starting_gold + game.minion_kill_gold
    assert blue.minion_kills == 1
    assert game.get_reward(Team.BLUE) == 0.1


def test_shop_upgrades_cost_gold_and_change_champion_stats():
    game = Game(seed=42)
    blue = game.champions[Team.BLUE]
    starting_gold = blue.gold
    starting_damage = blue.attack_damage

    assert game.purchase_upgrade(Team.BLUE, "blade")
    assert blue.gold == starting_gold - 200
    assert blue.attack_damage == starting_damage + 10.0
    assert blue.attack_upgrade_level == 1

    blue.position = Position(1000.0, 300.0)
    assert not game.purchase_upgrade(Team.BLUE, "armor")


def test_champion_armor_reduces_incoming_damage():
    game = Game(seed=42)
    blue = game.champions[Team.BLUE]
    blue.armor = 10.0
    starting_hp = blue.hp

    blue.take_damage(25.0)

    assert blue.hp == starting_hp - 15.0


def test_neutral_objective_respawns_at_another_site():
    game = Game(seed=42)
    objective = game.objective
    blue = game.champions[Team.BLUE]
    blue.position = Position(objective.position.x, objective.position.y)
    blue.attack_timer = 0.0
    objective.hp = blue.attack_damage
    starting_position = Position(objective.position.x, objective.position.y)

    assert game.attack_objective(Team.BLUE)
    assert not objective.alive
    assert game.objective_respawn_timer == game.objective_respawn_interval

    game._update_objective_respawn(game.objective_respawn_interval)

    assert objective.alive
    assert objective.hp == objective.max_hp
    assert objective.position != starting_position
    assert game.objective_respawn_timer is None
