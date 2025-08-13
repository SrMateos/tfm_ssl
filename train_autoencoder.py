import os
from src.config import ConfigParser
# Importa ambas clases
from src.training.trainer import Trainer

def main():
    config_parser = ConfigParser()
    config = config_parser.parse()

    trainer = Trainer(config)

    trainer.train()

if __name__ == "__main__":
    main()
