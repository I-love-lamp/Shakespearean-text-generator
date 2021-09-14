# Shakespearean-text-generator
This repository contains a python script that trains a RNN (Recurrent Neural Network) using text from Shakespeare's work and uses the model to predict a sequence of characters when provided with a sequence of characters.

## How this works
This is a text generator that uses a corpus consisting of an aggregation of Shakespeare's writings.
A RNN (Recurrent Neural Network) was trained on a character representation of the corpus.
The trained model was provided with a block of text (sequence of characters) and sequentially predicts the next character.
Given a sequence of characters from this data ("Alas! What dost tho"), train a model to predict the next character in the sequence ("u"). Longer sequences of text can be generated by calling the model repeatedly.

## How to run it
the starting point for the model is the next_char variable, a string constant. Set this as follows:
```next_char = tf.constant(['Your text here'])
