from nlp_id.tokenizer import Tokenizer 
from nlp_id.stopword import StopWord 

class Helper:
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        self.stopword = StopWord()
        self.punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    def tokenize(self, sentence): 
        return self.tokenizer.tokenize(sentence)

    def stopword_removal(self,sentence:str) -> str:
        return self.stopword.remove_stopword(sentence)
    
    def slang_cleaning(self, sentence, slang_df):
        words = self.tokenize(sentence)
        for i, word in enumerate(words):
            if word in slang_df['slang'].values:
                words[i] = slang_df.loc[slang_df['slang'] == word,'formal'].iloc[0]

        return " ".join(words)
    def remove_punctuations(self, sentence):
        return [word for word in sentence if word not in self.punctuation]        