
from src.dataloader import DataLoader
from src.simulation_base import SimulationBase


def main():
    data = DataLoader()
    sim = SimulationBase(data=data)
    sim.run()


if __name__ == '__main__':
    main()
