from lstm_chem.model import LSTMChem
from lstm_chem.dataloader import DataLoader
from lstm_chem.trainer import LSTMChemTrainer
from lstm_chem.generator import LSTMChemGenerator
from config import Config
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import IPythonConsole
RDLogger.DisableLog('rdApp.*')
import pandas as pd

config = Config()
modeler = LSTMChem(config, session='generate')
generator = LSTMChemGenerator(modeler)
print(config)

sample_number = 10000
sampled_smiles = generator.sample(num=sample_number)

valid_mols = []
for smi in sampled_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_mols.append(mol)
# low validity
print('Validity: ', f'{len(valid_mols) / sample_number:.2%}')

valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
# high uniqueness
print('Uniqueness: ', f'{len(set(valid_smiles)) / len(valid_smiles):.2%}')

# Of valid smiles generated, how many are truly original vs ocurring in the training data

training_data = pd.read_csv('./dataset_cleaned.smi', header=None)
training_set = set(list(training_data[0]))
original = []
for smile in valid_smiles:
    if not smile in training_set:
        original.append(smile)
print('Originality: ', f'{len(set(original)) / len(set(valid_smiles)):.2%}')


with open('./generations/gen0.smi', 'w') as f:
    for item in valid_smiles:
        f.write("%s\n" % item)