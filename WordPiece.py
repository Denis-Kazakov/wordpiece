import os, nltk, json
from collections import Counter
from tqdm import tqdm

class WordPieceTrainer:
    def __init__(self, vocab_size, valid_symbols=None, max_symbols=None):
        self.vocab_size = vocab_size
        
        # Initial set of symbols can be provided by the user or obtained from a text automatically
        self.valid_symbols = valid_symbols
        
        # Max number of symbols in the valid symbols lists.
        # This parameter is ignored if the user provides a list of symbols
        # Otherswise, only the max_number of most frequent symbols in the test will be used.
        # Others will become <unk>
        # If set to None, the list of symbols will not be truncated.
        self.max_symbols = max_symbols 
        
        self.history = []

    # MAIN FUNCTION
    
    def fit(self, text):
        """Build a list of tokens from a text using the WordPiece algorithm"""

        # Valid symbols
        if self.valid_symbols is None:
            self.valid_symbols, invalid_symbols = self._get_symbols(text, self.max_symbols)
        # Valid characters are the starting point for building the token vocabulary
        self.tokens = self.valid_symbols

        # Replace all symbols that are not valid with <unk>
        if not invalid_symbols:
            for symb in invalid_symbols:
                text.replace(symb, '<unk>')
        
        # Get a list of unique words (split into tokens)
        word_counts = [{'word': self._split_word(word), 'count': count} for word, count in nltk.FreqDist(text.split()).items()]

        num_tokens = len(self.tokens)
        while num_tokens < self.vocab_size:
            unigram_counts, bigram_counts = self._count_ngrams(word_counts)
    
            # Mutual information (almost): 
            #bigram probability (frequency) divided by the product of its elements' probabilities.
            # Actually, I used counts insteand of frequencies because 
            # I am only need to compare relative values and the totals are the same for all bigrams and unigrams
            bigram_mi = []
            for bigram, count in bigram_counts.items():
                bigram_mi.append(
                    {
                        'bigram': bigram,
                        'mi': count / (unigram_counts[bigram[0]] * unigram_counts[bigram[1]])
                    }
                )
            bigram_mi.sort(key=lambda x: x['mi'], reverse=True)
    
            # Several bigrams can be merged into new tokens in a single pass if those bigrams are independend,
            # i.e. do not include identical unigrams
            bigrams_to_merge, new_tokens = self._get_independed_bigrams(bigram_mi)
            self.history.extend(bigrams_to_merge)
            self.tokens.extend(new_tokens)
            for element in word_counts:
                element['word'] = self._merge_bigrams_in_word(element['word'], bigrams_to_merge)
            num_tokens = len(self.tokens)
            print(f"Total tokens in vocab: {num_tokens}")

        if '<unk>' not in self.tokens:
            self.tokens = ['<unk>'] + self.tokens
        if '<mask>' not in self.tokens:
            self.tokens = ['<mask>'] + self.tokens
        if '<sep>' not in self.tokens: # Checking just in case
            self.tokens = ['<sep>'] + self.tokens
        if '<cls>' not in self.tokens:
            self.tokens = ['<cls>'] + self.tokens
        if '<pad>' not in self.tokens: 
            self.tokens = ['<pad>'] + self.tokens
        with open('./tokenizer_data/idx2token.json', 'w') as f:
                json.dump(self.tokens, f)

        token2ix = {token: number for number, token in enumerate(self.tokens)}
        with open('./tokenizer_data/token2idx.json', 'w') as f:
                json.dump(token2ix, f)        
        return self.tokens

    # HELPER FUNCTIONS
        
    def _get_symbols(self, text, max_symbols):
        if max_symbols is None:
            valid_symbols = list(set(text))
            invalid_symbols = []
        else:
            all_symbols = set(text)
            valid_symbols = nltk.FreqDist.most_common(max_symbols)
            valid_symbols = [item[0] for item in valid_symbols]
            invalid_symbols = list(all_symbols - set(valid_symbols))

        # The beginning-of-word character in WordPiece: 
        if '_' not in valid_symbols:    
            valid_symbols.append('_')

        with open('./tokenizer_data/valid_symbols.json', 'w') as f:
                json.dump(valid_symbols, f)

        return valid_symbols, invalid_symbols
    
    def _split_word(self, word: str) -> list:
        """Split word into a list of characters with the first character prepended with '_'. 
        <unk> is merged into a single list element.
        A word is any sequence of characters without whitespace."""
        word = ' '.join(list(word)).replace('< u n k >', '<unk>')
        return ['_'] + word.split()

    def _count_multiply(self, seq: list, count: int) -> dict:
        """Count elements in a seq and multiply their counts by the seq count in the text"""
        c = Counter(seq)
        return {key: value * count for key, value in c.items()}

    def _count_ngrams(self, word_counts: list) -> tuple:
        """Count bigrams and unigrams allowing for the count of each word in the sequence.
        word_counts: a list of dicts with a split word and its count in the text
        output: a tuple of two counter objects (unigrams and bigrams)"""
        unigram_counts = Counter()
        bigram_counts = Counter()
        for element in word_counts:
            unigram_counts.update(
                self._count_multiply(element['word'], element['count'])
            )
            bigram_counts.update(
                self._count_multiply(nltk.bigrams(element['word']), element['count'])
            )
        return unigram_counts, bigram_counts

    def _get_independed_bigrams(self, bigram_mi):
        """
        Input: ranked list of bigrams
        Output: top bigrams that do not include same unigrams
        """
        bigrams_to_merge = []
        unique_unigrams = []
        new_tokens = []
        for element in bigram_mi:
            bigram = element['bigram']
            if bigram[0] not in unique_unigrams and bigram[1] not in unique_unigrams:
                bigrams_to_merge.append(bigram)
                new_tokens.append(''.join(bigram))
                unique_unigrams.extend(bigram)
            else:
                break
        return bigrams_to_merge, new_tokens

    def _merge_bigrams_in_word(self, word: list, bigrams_to_merge: list) -> list:
        word = ' '.join(word)
        for bigram in bigrams_to_merge:
            word = word.replace(
                bigram[0] + ' ' + bigram[1],
                bigram[0] + bigram[1]
            )
        return(word.split())


class WordPieceTokenizer:
    def __init__(self, path='./tokenizer_data/'):
        with open(os.path.join(path, 'valid_symbols.json')) as f:
            self.valid_symbols = json.load(f)
        with open(os.path.join(path, 'idx2token.json')) as f:
            self.idx2token = json.load(f) # List of tokens
        with open(os.path.join(path, 'token2idx.json')) as f:
            self.token2idx = json.load(f)

    # MAIN FUNCTIONS
    
    def encode(self, text: str, display_progress_bar=True) -> list:
        """Convert a text into a list of token indices"""
        # Replace unknown symbols with <unk>
        all_symbols = set(text)
        invalid_symbols = list(all_symbols - set(self.valid_symbols))
        if not invalid_symbols:
            for symb in invalid_symbols:
                text.replace(symb, '<unk>')
        
        # Tokenizing the text
        output = []
        words = ['_' + word for word in text.split()]
        if display_progress_bar:
            for word in tqdm(words):
                output.extend(self._tokenize_word(word))
        else:
            for word in words:
                output.extend(self._tokenize_word(word))
            
        # Convert tokens into token indices
        return [self.token2idx[token] for token in output]

    
    def decode(self, sequence: list) -> str:
        """Convert a sequence of token indices into text"""
        text = ''.join(
            [self.idx2token[idx] for idx in sequence]
        )
        return text.replace('_', ' ')

    # HELPER FUNCTIONS
    
    def _tokenize_word(self, word):
        cur_output = []
        start, end = 0, len(word)
        # Segment token with the longest possible subwords from symbols
        while start < len(word) and start < end:
            if word[start: end] in self.idx2token:
                cur_output.append(word[start: end])
                start = end
                end = len(word)
            else:
                end -= 1
        if start < len(word): # Should not happen but just in case
            cur_output.append('<unk>')
        return cur_output
        