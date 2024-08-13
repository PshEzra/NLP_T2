import pickle
import numpy as np
from aksara import BaseTokenizer

class Preprocess:
  def __init__(self) -> None:
    self.data: list[str] = None
    self.train_data: list[str] = None
    self.test_data: list[str] = None
  
  """
  - Parameter path merupakan lokasi data diletakkan.
  - Data disarankan untuk disimpan di dalam folder data yang telah disediakan
  - Di dalam folder tersebut tersedia tiga jenis data, yakni idwiki-corpus.txt, idwiki-train.txt, dan idwiki-test.txt
  - Rasio antara train:test yang terdapat pada file idwiki-train.txt dan idwiki-test.txt merupakan 80:20
  - idwiki-corpus.txt merupakan gabungan antara idwiki-train.txt dan idwiki-test.txt
  - idwiki-corpus.txt disediakan untuk Anda yang ingin bereksperimen dengan rasio train:test atau sejenisnya.
  """
  def load_data(self, path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
      data = f.readlines()
    
    data_type = path.split('/')[-1].split('-')[-1].split('.')[0]
    if data_type == 'train':
      self.train_data = data
    elif data_type == 'test':
      self.test_data = data

  def get_train_data(self) -> list[str]:
    return self.train_data
  
  def get_test_data(self) -> list[str]:
    return self.test_data
  
  """
  - Fungsionalitas method di bawah ini adalah melakukan pembersihan kalimat dari simbol newline ('\n').
  - Di samping itu, method diharapkan dapat melakukan trimming whitespace pada awal dan akhir kalimat.
  - Output format yang diharapkan berupa list of strings.
  """
  def split_sentences(self, data: list[str]) -> list[str]:
    # TODO: Implement based on the given description
    ret_list = []

    for i in data:
      ret_list.append(i.replace('\n', '').strip())
    
    return ret_list
  
  """
  - Fungsionalitas method di bawah ini adalah melakukan tokenisasi pada kalimat.
  - Sebelum mengerjakan method ini, pastikan Anda telah melakukan eksplorasi pada dataset.
  - Perlu diperhatikan terdapat dua macam parameter yang disediakan, yakni data dan lower. Parameter lower digunakan untuk melakukan eksperimen lowercasing dan non-lowercasing.
  - Character/kata pada dataset tidak seluruhnya ascii/latin. Lakukan penanganan/filtering terhadap character/kata non-ascii.
  - Perlu diperhatikan bahwa n-gram merupakan case-sensitive model. Kata "saya" dan "Saya" merupakan dua hal yang berbeda. Silakan Anda tentukan penanganan yang Anda rasa terbaik.
  - Output format yang diharapkan adalah list of lists of strings.
  """
  def tokenize_sentences(self, data: list[str], lower: bool) -> list[list[str]]:
    # TODO: Implement based on the given description
    tokenized_list = []

    tokenizer = BaseTokenizer()

    for sentence in data:
      sen = sentence.encode('ascii', 'ignore').decode('ascii')
      if lower:
        sen = sen.lower()
      sentence_list = tokenizer.tokenize(sen)
      tokenized_list += sentence_list

    return tokenized_list
    pass
  
  def get_tokenized_data(self, data: list[str], lower: bool) -> list[list[str]]:
    splitted: list[str] = self.split_sentences(data)
    tokenized: list[str] = self.tokenize_sentences(splitted, lower)
    return tokenized

  """
  - Fungsionalitas pada method di bawah ini adalah menghitung kemunculan kata di dalam corpus.
  - Output format yang diharapkan berupa dictionary dengan pasangan key berupa kata/token dan value berupa jumlah kemunculan kata/token tsb.
  """
  def word_map(self, data: list[list[str]]) -> dict:
    # TODO: Implement based on the given description
    ret_dict = {}

    # print(data)
    for sentence in data:
      for word in sentence:
        if word not in ret_dict.keys():
          ret_dict[word] = 1
        else:
          ret_dict[word] += 1

    return ret_dict
    pass
  
  """
  - Fungsionalitas pada method di bawah ini adalah melakukan filtering terhadap kata yang kemunculannya di bawah threshold/batasan tertentu.
  - Misalkan Anda menetapkan setiap kata yang kemunculannya di bawah 5 tidak perlu diproses (threshold = 5).
  - Anda diharapkan memanfaatkan fungsionalitas method word_map yang telah diimplementasikan sebelumnya.
  - Expected output berupa kumpulan kata yang kemunculannya di atas atau sama dengan (greater than or equal to) threshold.
  - Output format berupa list of strings.
  """
  def filter_vocab_by_threshold(self, data: list[list[str]], num_threshold: int) -> list[str]:
    # TODO: Implement based on the given description
    word_dict: dict = Preprocess.word_map(Preprocess, data)
    new_data = []

    for word in word_dict.keys():
      if word_dict[word] >= num_threshold:
        new_data.append(word)

    # for sentence in data:
   
    #  for word in sentence:
    #    if word not in word_dict.keys():
    #      new_data.append(word)
    #    else:
    #      if word_dict[word] >= num_threshold:
    #        new_data.append(word)

    return new_data
  
  """
  - Fungsionalitas pada method ini adalah mengganti kata-kata yang kemunculannya di bawah threshold menjadi simbol <unk>.
  - Simbol <unk> melambangkan kata tersebut merupakan OOV (Out of Vocabulary).
  - Anda diharapkan memahami alur method ini dengan memanfaatkan method preprocess_raw_data.
  - Output format berupa list of lists of strings.
  """
  def handle_oov_with_unk(self, data: list[list[str]], vocab: list[str], unknown_token='<unk>') -> list[list[str]]:
    # TODO: Implement based on the given description
    new_data = []

    for sentence in data:
      new_words = []

      for word in sentence:
        if word in vocab:
          new_words.append(word)
        else:
          new_words.append(unknown_token)
      
      new_data.append(new_words)

    return new_data
  
  def preprocess_raw_data(self, train, test, threshold):
    vocab = self.filter_vocab_by_threshold(train, threshold)
    train_handled = self.handle_oov_with_unk(train, vocab)
    test_handled = self.handle_oov_with_unk(test, vocab)
    return vocab, train_handled, test_handled
  
  def save_to_pickle(self, vocab: list[str], train: list[list[str]], test: list[list[str]], lower: bool) -> None:
    filename = {'vocab': vocab, 'train': train, 'test': test}
    for key, value in filename.items():
      filename = f'./test-data/idwiki-{key}-uncased.pkl'
      if lower:
        filename = f'./test-data/idwiki-{key}-cased.pkl'
      with open(filename, 'wb+') as f:
        pickle.dump(value, f)
  
  def load_from_pickle(self, lower: bool):
    suffix = 'uncased'
    if lower:
      suffix = 'cased'
    with open(f'./test-data/idwiki-vocab-{suffix}.pkl', 'rb') as f:
      vocab = pickle.load(f)
    with open(f'./test-data/idwiki-train-{suffix}.pkl', 'rb') as f:
      train = pickle.load(f)
    with open(f'./test-data/idwiki-test-{suffix}.pkl', 'rb') as f:
      test = pickle.load(f)
    return vocab, train, test
  
def main():
  preprocess = Preprocess()

  print('Loading data...')
  preprocess.load_data('./test-data/idwiki-train.txt')
  preprocess.load_data('./test-data/idwiki-test.txt')

  train = preprocess.get_train_data()
  test = preprocess.get_test_data()

  """
  EXAMPLE:
  - Pada contoh ini menggunakan skenario no lowercasing (cased)
  """
  lowercase: bool = False
  tokenized_train = preprocess.get_tokenized_data(train, lowercase)
  tokenized_test = preprocess.get_tokenized_data(test, lowercase)

  """
  EXAMPLE:
  - Pada contoh ini ditetapkan threshold sebesar 3.
  - Artinya setiap kata yang kemunculannya di bawah 3 akan dihilangkan dari koleksi.
  """
  print('Pre-processing data...')
  vocab, train_handled, test_handled = preprocess.preprocess_raw_data(tokenized_train, tokenized_test, 3)

  print('Saving data...')
  preprocess.save_to_pickle(vocab, train_handled, test_handled, lowercase)
  print('Finish!')

if __name__ == "__main__":
  main()
