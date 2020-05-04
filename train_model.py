from lstm_chem.model import LSTMChem
from lstm_chem.dataloader import DataLoader
from lstm_chem.trainer import LSTMChemTrainer
from config import Config
from copy import copy


config = Config()
model = LSTMChem(config,'train')
train_dataloader = DataLoader(config,'train')
valid_dataloader = copy(train_dataloader)
valid_dataloader.data_type = 'valid'
trainer = LSTMChemTrainer(model,train_dataloader,valid_dataloader)
trainer.train()
trainer.model.save_weights('./code_exp/LSTM_Chem/checkpoints/LSTM_Chem-baseline-model-full.hdf5')