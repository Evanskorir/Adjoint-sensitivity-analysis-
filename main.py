from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    data = DataLoader(model="british_columbia")
    runner = Runner(data=data, model="british_columbia")
    runner.run()


if __name__ == '__main__':
    main()
