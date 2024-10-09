from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    data = DataLoader(model="rost")
    runner = Runner(data=data, model="rost")
    runner.run()


if __name__ == '__main__':
    main()
