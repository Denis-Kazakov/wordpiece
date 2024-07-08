# WordPiece tokenizer for BERT
A simple implementation to train a WordPiece tokenizer on a large txt file for use with BERT. The Jupyter notebook uses the WikiText-2 train dataset (saved as a txt file) as an example.

WordPiece.py is a library with two classes:
- WordPieceTrainer to train a tokenizer. Training output to be used to tokenized new data is stored in the /tokenizer_data/ folder (in this case, tokens obtained from the WikiText-2 dataset).
- WordPieceTokenizer to use a trained tokenizer.
