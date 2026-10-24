from .game import Game


def main() -> None:

    game = Game(seed=42)

    print("================================")
    print("          MOBA-RL")
    print("      Simulation Engine")
    print("================================")

    print("\nStarting simulation...")

    for _ in range(100):

        game.step(dt=0.1)

        if game.tick_count % 10 == 0:
            game.print_state()

    print("\nSimulation finished.")

    if game.winner is not None:
        print(
            f"Winner: {game.winner.value.upper()}"
        )
    else:
        print("Winner: None")


if __name__ == "__main__":
    main()