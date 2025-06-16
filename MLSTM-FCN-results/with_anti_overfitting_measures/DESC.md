# Initial results for MLSTM-FCN architecture
## Model
* This is an iteration on the model from MLSTM-FCN paper, with added anti-overfitting measures in hope that it will better generalize
* Specific architecture in .ipynb file
## Training
* Trained and validated on "validation" data (val split 0.2)
* Initial results after 50 epochs on Athena
## Notes
* Does not want to generalize from train to validation, tops at about 43% acc
* Plateaus at 93% even during further training (results soon)
