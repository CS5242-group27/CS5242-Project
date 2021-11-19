# CS5242-Project by Group 27

cd src

## MLP
### Basic MLP-------mlp.py
### Improved MLP-------mlp_improve.py
1. Modify the dataset path in train() and test()
2. The accuracy and loss plot can be found in log/ by using Tensorboard
3. Models will be saved in models/
4. You can also adjust the size of each frame in class MyDataset()

Then run 'python [filname].py'

## CNNs. 

Logs and models would be saved in ../log and ../models respectively.

cd cnn
Open cnn.ipynb, and then run each cell in order.
cd -


## RNN
### Basic RNN-------rnn.py
### Improved RNN-------rnn_improve.py
1. Modify the dataset path in train() and test()
2. The accuracy and loss plot can be found in log/ by using Tensorboard
3. Models will be saved in models/
4. You can also adjust the size of each frame in class MyDataset()

Then run 'python [filname].py'

### ANN

1. Create ann/log and ann/models folders.
2. Open ann/main.py, change the value of global variable "MODEL_NAME" to the name you want. Names start with "basic_" will be a basic ANN; names start with "improve_" will create a improved ANN.
3. python main.py    (to train the model)
4. After the training finish, comment the "train(model_name)" in main() function and uncomment the "test(model_path)" sentence. 
5. python main.py    (to test the model)

