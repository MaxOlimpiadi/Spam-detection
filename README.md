# Spam SMS Classifier (Logistic Regression from Scratch)

Minimal implementation of a **binary spam classifier** using **logistic regression** trained with **batch gradient descent**.  
The model is trained from raw text (one message per line) and uses a simple **bag-of-words (term frequency)** representation.

## What it does
- Loads `SMS_Spam_Corpus_big.txt` (labels: `spam` / `ham` at end of each line)
- Tokenizes and lemmatizes messages with **spaCy** (`en_core_web_sm`)
- Builds a vocabulary and vectorizes each message into term-frequency features
- Trains logistic regression with gradient descent
- Stops when the loss (log-loss) stops improving

