This is a simple old school convolutional neural network which if specified can train every convolutional layer independently on the labelled data, such that in essence three models are being trained. The first model would be comprised of a single conv layer, a dense layer and a softmax activation. Once the first model has been trained the value of the weights from the conv layer are insterted into the first layer weights of the second model which has two conv layers, a dense layer and then a softmax activation. The second model is trained such that the previously inserted weights are frozen. The first and second layer weights are collected and inserted into a third model comprised of two conv layers two dense layers and softmax. The weights are inserted as the weight values of the third model then frozen while the rest of the weights from above layers are adjusted during training. The accuracies, as well as the train and test neuron values from the model are then collected for mapping function smoothness between layers and labels later on. 

Run with python3

To test run:

python app.py --model_type='irregular' --epochs=2 --batch_size=128 --num_epochs_per_decay=350. --lr_decay_factor=0.1 --dev=True

To do:

-unhack tensorflow data method use in data.py
-allow frozen weights to adjust with varying learning rates
-add logging and exceptions
-write layers in compressed file to speed things up
-integrate with waveletforest code
