#### Description:

This is a shallow convolutional autoencoder. The layers are:
            input -->  convolutional --> max pool1 --> dropout1 --> fully connected1 --> 
            dropout2 -->  fully connected2 --> (internal layer, lower dimenisonal representation) -->
            fully connected3 --> dropout3 --> fully connected4 --> dropout4 --> deconvolutional1 --> 
            upsample1 --> fully connected4 (outputlayer) 

#### Usage: 

python convolutional_autoencoder.py



