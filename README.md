# Corona Drug Discovery


## Introduction
This is a solution for the Possible Drugs for **Covid-19** . Binding scores of leading existing drugs (HIV inhibitors) are around -10 to -11  and around -13 for the drug Remdesivir which recently entered clinical testing. More negative the binding score is, better the drug is. The goal is to create a novel small molecule which can bind with the coronavirus, using deep learning techniques for molecule generation and *PyRx* to evaluate binding affinities. By combining a **Generative RNN** model with techniques and principles from *transfer learning* and *genetic algorithms*, I was able to create several small molecule candidates which achieved binding scores approaching -18.

## Data Preparation

The original network only trained on ~450k unique **SMILES**. My first goal was to train a network from scratch that would be highly adept at generating robust, realistic molecules.

I combined data sets from two source:

 1. [Moses Data Set](https://github.com/molecularsets/moses)
 2. [ChEMBL Data Set](https://www.ebi.ac.uk/chembl/)
 
Together these two data sets represented about 4 millions SMILES. After cleaning the SMILES using the [clean_smiles.py](https://github.com/Luckygyana/Corona-Drug-Discovery/blob/master/clean_smiles.py) script and only retaining SMILES between 34 to 128 characters in length, [./dataset_cleaned.smi](https://github.com/Luckygyana/Corona-Drug-Discovery/blob/master/dataset_cleaned.smi) contains the final list of ~4.8 million smiles on which the initial network was trained.

 
## Model Architecture
![all text](https://onlinelibrary.wiley.com/cms/asset/e8c33e80-1633-4bfc-86da-7078c633b74c/minf201700111-toc-0001-m.jpg)
Schematic of model training(**left**) and compund design by sampling(**right**)




![all text](https://miro.medium.com/max/670/1*oa8X-Rn9AtmO2ZBEg76DTg.jpeg)

## Model Summary 
Model: Sequential
| Layer(type)      | Output Shape     | Parameters    |
|------------------|:-------------------:|----------:|
| lstm(LSTM)       | (None,None,256)   | 316416       |
| lstm1(LSTM)     | (None,None,256) |525312 |
| dense(Dense)    |(None,None,52)  |13364  |

Total parameters: 855092

Trainable parameters: 855092

Non-Trainable parameters: 0

## Generating Initial Universe of SMILES

After completing training, I used the new network to generate **10,000 SMILES**. I would have liked to generate more to start with a wider set of molecules to evaluate before narrowing in on molecules that bind well, but time was a constraint as the generation process takes several hours.

I evaluated relative performance of the original repository network vs my new network along two metrics from the original repository, and added a third metric of my own creation:
  1.  **Validity**: of the total number of generated smiles, percentage that are actually valid smiles for molecules
  2.  **Uniqueness**: of the total valid number of generated smiles, percentage that are not duplicates 
  3.  **Originality**: of the total number of valid generated smiles, percentage that are brand new creations that do not appear in the training data.
 
 Original LSTM_Chem network benchmarks: 
 1. *Validity*: 62.3% 
 2. *Uniqueness*: 99.8% 

My newly trained network metrics: 
1. *Validity*: 97.0% 
2. *Uniqueness*: 99.8% 
3. *Originality*: 89.0% 

Originally generated 100 smiles are saved to [./generations0.smi](https://github.com/Luckygyana/Corona-Drug-Discovery/blob/master/generations0.smi)

## Finding Top Candidates from Initial Universe of SMILES

Having generated ~10k new valid molecules, my biggest constraint was time: evaluating each molecule's binding affinity with the *coronavirus protease* via *PyRx* is a lengthy process with ~1.5 molecules evaluating per minute. Running an analysis of 10k molecules was therefore not possible as that would take over 100 hours. In order to minimize time the function **initialize_generation_from_mols()** randomly picks 30 molecules, then iterates through the rest of the list calculating that molecules **Tanimoto Similarity** scores to the molecules so far added to the list, and only 'accepts' the molecule if the maximum similarity score is less than a certain threshold. This ensures that even a smaller sample will feature a diverse set of molecules. I was then able to save smiles to a pandas dataframe (csv), and also convert smiles to molecules and write these molecules to an *SDF* file which could be manually imported into *PyRx* for analysis. *PyRx* then outputs a *csv* of molecules and their binding scores after analysis. In order to relate smiles in my *pandas/csv* to molecules as *SDF* in *PyRx,* I used **Chem.PropertyMol.PropertyMol(mol).setProp()** to set the 'Title' property to a unique identifier of four letters and a generation number.


## Transfer Learning & Genetic Algorithm

After evaluating about **1500 gen0 SMILES** in *PyRx* (an overnight task), I had a variety of scores for a diverse set of molecules. I then employed techniques and principles of genetic algorithms and transfer learning to take the original network's knowledge of creating realistic molecules and transfer it to the domain of making molecules specifically adept at binding with the coronavirus.

Each generation I ran the following steps:

 1. Ranked all molecules so far tested across all generations by binding scores, and picked the top X smiles with the best scores (**I used 35**).
 2. Calculate the similarity of each remaining molecule with the set of molecules from Step 1, and calculate an adjusted score that boosts molecules that are very different from the top ranking molecules and have good but not great scores (i.e. they may work via a different mechanism so keep exploring that mechanism). Take the top X smiles ranked by this similarity adjusted score (**I used 5**).
 3. In basic research, I learned that one of the most critical defining characteristics of small molecules is that they weigh **less than 900 daltons**. I noticed that larger molecules (1000-1050 range) seemed to be getting high binding affinity scores, so in order to both learn from what made those large molecules good, but also promote smaller molecules, I computed a weight adjusted score that boosts light weight molecules with good but not great scores. I then ranked by this adjusted score and took the top X molecules (I used 5).
 4. The above steps yielded a list of *45 molecules* considered 'good fits' across three metrics of fit: 
	 **i)** overall score, 
	 **ii)** similarity adjusted score (ensuring diverse molecules are included), 
	**iii)** weight adjusted score (ensuring especially small molecules are included).
	 As a way to promote random 'mutations' (inspired by a genetic algorithm approach) I used the baseline model to generate a random sample of molecules in each generation. (I took 5)
  5. Now we have 50 total 'target' SMILES (i.e. the 'parents' for our genetic algorithm). I then added the previous generation's network on these 50 target SMILES. I applied a **rule-of-thumb** to train the network enough to cut its loss in half from the first epoch to the last epoch each time. By trial-and-error, I found this to typically be ~5 epochs, so that's what I used. 
  6. After training a new model on the generation's well-fit "parents", I used it to generate the next generation of ideally better-fit "children". As a rule of thumb I would generate 500 SMILES each generation, which after removing duplicates, invalids, etc usually led to a few hundred children to evaluate per generation.
  7. Save the new generation to the tracking *pandas/csv* and to molecule *SDF*, then feed *SDF* into *PyRx* and evaluate.


I repeated the above steps across 10 generations, each time using the best fit + 'mutation' training set from the prior generation to teach the network to create molecules better and better at binding

## Next Steps

 
1. Have a domain expert analyze top findings for fit and/or find the molecules in the universe of existing approved drugs which are most similar to top findings and evaluate them for fit.
 2. According to this [paper](https://arxiv.org/pdf/1703.07076.pdf), the baseline model may be further improved by training on a universe of enumerated SMILES, not just canonical SMILES.
  3. Code needs refactoring.

