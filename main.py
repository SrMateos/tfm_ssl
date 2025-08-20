from src.config import ConfigParser
from src.training.trainer import Trainer

def main():
    config_parser = ConfigParser()
    config = config_parser.parse()

    trainer = Trainer(config)

    trainer.train()

if __name__ == "__main__":
    main()
