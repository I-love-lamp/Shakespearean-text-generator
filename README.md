# Random Shakespearean text generator
This repository contains a python script that trains a RNN (Recurrent Neural Network) using text from Shakespeare's work and uses the model to predict a sequence of characters when provided with a sequence of characters.

## How this works
This is a text generator that uses a corpus consisting of an aggregation of Shakespeare's writings.
A RNN (Recurrent Neural Network) was trained on a character representation of the corpus.
The trained model was provided with a block of text (sequence of characters) and sequentially predicts the next character.
Given a sequence of characters from this data ("Alas! What dost tho"), train a model to predict the next character in the sequence ("u"). Longer sequences of text can be generated by calling the model repeatedly.

## How to run it
the starting point for the model is the next_char variable, a string constant. Set this as follows:  
```next_char = tf.constant(['Your text here'])```  
For example, providing this text **Alas!** gave a result like this:  
> hat dost thou, pretty, rather
To touch this speech with you and yours becomes.
KING EDWARD IV:
Hold, very well, good morrow; Catesby, for myself; but tell these arm
We all rest friends o' the comfort of thy sirs,
Were from my horse; and all the remedy is
required, or ends in the vice: and his smother
Mie-bediens shook off, and countenance, sour
very fined, rather, that I have tush'd your eyes
With distemper'd weakness laughter:
Will thither by goddess Clearing, hath been there
now that thou wear'sty! stand up, starve,
To tell the sumples of the king.  

The response varies each time the script is run.
