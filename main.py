from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    data = DataLoader()
    runner = Runner(data=data)
    runner.run()


if __name__ == '__main__':
    main()
