import hydra
from transformers import AutoModelForQuestionAnswering
from utils.util import set_seed
from trainers.trainer import Trainer
from data_loader.data_loaders import DataLoader
import torch
from datetime import datetime
# export HYDRA_FULL_ERROR=1


@hydra.main(version_base="2.5", config_path=".", config_name="config.yaml")
def main(config):
    # seed
    set_seed(config.seed)

    model = AutoModelForQuestionAnswering.from_pretrained(config.checkpoint)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=config.lr)

    data_loader = DataLoader(config)

    trainer = Trainer(model,
                      optimizer,
                      config=config,
                      train_data_loader=data_loader.train_data_loader,
                      valid_data_loader=data_loader.valid_data_loader,
                      device=config.device
                      )

    trainer.train()
    torch.save(model.state_dict(), 'saved/'+config.model+str(datetime.now())+'.pt')
    #model.save_pretrained("./models/bert-base-cased/")
    # trainer.validate()
    # trainer.inference(10, config)


if __name__ == '__main__':
    main()
