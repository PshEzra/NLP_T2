import math
import preprocess_data
from itertools import chain

class NgramModel:
  def __init__(self, vocab: list[str], train: list[list[str]], test: list[list[str]]) -> None:
    self.vocab = vocab
    self.train = train
    self.test = test
  
  """
  - Fungsionalitas model ini adalah menghasilkan koleksi n-gram model.
  - Seperti 2-gram, artinya di dalam koleksi terdapat pasangan, seperti: '<s> saya', 'saya sedang', 'sedang makan', 'makan nasi', 'nasi kapau'.
  - Expected output berupa dictionary dengan pasangan key berupa n-length token serta value berupa kemunculan n-length token tersebut di dalam corpus.
  - Output format berupa dictionary.
  """
  def generate_n_grams(self, data: list[list[str]], n: int, start_token: str = '<s>', end_token='</s>') -> dict:
    # TODO: Implement based on the given description
    ret_dict = {}

    # print(data)
    for sentence in data:
      new_sentence = sentence
      if new_sentence[-1] != end_token:
        new_sentence.insert(len(new_sentence), end_token)

      if new_sentence[0] != start_token:
        new_sentence.insert(0, start_token)
      
      for i in range(0, len(new_sentence)-n):
        
        inserted_string = []
        for j in range(n):
          inserted_string.append(new_sentence[i+j])
        new_str = " ".join(inserted_string)

        if new_str not in ret_dict.keys():
          ret_dict[new_str] = 1
        else:
          ret_dict[new_str] += 1
    
    return ret_dict
  
  """
  - Fungsionalitas method ini menghitung probabilitas suatu kata given kata/kumpulan kata.
  - Sederhananya, method ini merupakan implementasi dari ekspresi P(w_i|w_1:{i-1}).
  - Perlu diperhatikan bahwa pada parameter terdapat 'laplace_number' yang artinya Anda diharapkan mengimplementasikan add-one (laplace) smoothing.
  - Output format berupa float.
  """
  def count_probability(self, predicted_word: str, given_word: list[str], n_gram_counts, n_plus1_gram_counts, vocabulary_size, laplace_number: float = 1.0) -> float:
    # TODO: Implement based on the given description
    # print(given_word)

    end_amount = 0
    if isinstance(n_gram_counts, list):
      if predicted_word in n_plus1_gram_counts.keys():
        amount = n_plus1_gram_counts[predicted_word]
      else:
        amount = 0
  
      if given_word[0] in n_plus1_gram_counts.keys():
       amount_other = n_plus1_gram_counts[given_word[0]]
      else:
       amount_other = 0

      divider = 0

      for sentence in n_gram_counts:
        divider += len(sentence)
      
      # print(vocabulary_size)

      end_amount = ((amount+1) / (divider+vocabulary_size))#*((amount_other+1) / (divider+vocabulary_size))
      # print(divider, vo)
      # print("Unigram", amount, divider)
      # print(amount / divider)
      return end_amount
    
    else:
      given_join = " ".join(given_word)
      new_string = " ".join(list(given_word) + [predicted_word])
      # print(new_string)
      amount = 0
      divider = 0

      if new_string in n_plus1_gram_counts.keys():
        amount = n_plus1_gram_counts[new_string] 
      if given_join in n_gram_counts.keys():
        divider = n_gram_counts[" ".join(given_word)]

      end_amount = (amount+1) / (divider+vocabulary_size)
      # print("NGram", divider)
      # print(n_gram_counts[" ".join(given_word)])
      # print(new_string, amount)

    return end_amount
  
  """
  - Silakan Anda menggunakan method ini untuk bermain-main/menguji segala kemungkinan sentence/word generation berdasarkan method count_probability yang telah Anda bangun.
  """
  def probabilities_for_all_vocab(self, given_word: list[str], n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='</s>', unknown_token='<unk>',  laplace_number=1.0):
    given_word = tuple(given_word)
    vocabulary = vocabulary + [end_token, unknown_token]
    vocab_size = len(vocabulary)

    probs = dict()
    for word in vocabulary:
      prob = self.count_probability(word, given_word, n_gram_counts, n_plus1_gram_counts, vocab_size, laplace_number=laplace_number)
      probs[word] = prob
    probs['<unk>'] = -1
    return probs

  """
  - Fungsionalitas pada method ini adalah untuk mengevaluasi n-gram model Anda menggunakan metrik perplexity.
  """
  def count_perplexity(self, sentence, n_gram_counts, n_plus1_gram_counts, vocab_size, start_token='<s>', end_token = '</s>', laplace_number=1.0):
    # TODO: Implement based on the given description
    perplexity_ret = 0

    new_sentence = sentence
    if new_sentence[-1] != end_token:
      new_sentence.insert(len(new_sentence), end_token)

    if new_sentence[0] != start_token:
      new_sentence.insert(0, start_token)
    
    m = 0
    n = 0

    if isinstance(n_gram_counts, list):
      n = 1
    else:
      n = len(list(n_gram_counts.keys())[0].split())

    for i in range(n, len(new_sentence)):
      words = new_sentence[i-n:i+1]
      num = self.count_probability(words[-1], words[:-1], n_gram_counts, n_plus1_gram_counts, vocab_size, laplace_number)
      m += 1
      perplexity_ret += math.log(num, 2)
    # print("Prob", perplexity_ret)
    perplexity_ret = perplexity_ret / (-m)
    return perplexity_ret

def main():
  """
  EXAMPLE:
  - Pada contoh ini menggunakan scenario no lowercasing (cased)
  """
  lowercase: bool = False
  preprocess = preprocess_data.Preprocess()
  vocab, train, test = preprocess.load_from_pickle(lowercase)

  model = NgramModel(vocab, train, test)
  flatten_test = list(chain.from_iterable(test))

  """
  EXAMPLE:
  - Silakan berkreasi se-kreatif mungkin menggunakan beragam n-gram model yang Anda inginkan.
  - Anda dibebaskan untuk mengganti/menambah/menghapus contoh kombinasi di bawah ini sesuai dengan kreativitas Anda.
  - Kami sangat menghargai kreativitas Anda terkait Tugas Individu 2 ini.
  """
  print('\nGENERATE N GRAM\n')

  unigram_counts = model.generate_n_grams(train, 1)
  bigram_counts = model.generate_n_grams(train, 2)

  print("Unigram Vocab :", len(unigram_counts.keys()))
  print("Bigram Vocab :", len(bigram_counts.keys()))

  trigram_counts = model.generate_n_grams(train, 3)

  """
  EXAMPLE:
  - Di bawah ini merupakan contoh/cara untuk generate kalimat dari n-gram LM
  """

  """
  UNIGRAM
  """
  print("\nUNIGRAM")
  generate_S_unigram = model.probabilities_for_all_vocab(['karena'], train, unigram_counts, vocab)
  # print(max(generate_S_unigram, key=generate_S_unigram.get))
  # print(sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[:5])

  word = ['karena']
  for i in range(10):
    generate_S_unigram = model.probabilities_for_all_vocab([word[-1]], train, unigram_counts, vocab)
    next_word = max(generate_S_unigram, key=generate_S_unigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print( " ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['Berkas']
  for i in range(10):
    generate_S_unigram = model.probabilities_for_all_vocab([word[-1]], train, unigram_counts, vocab)
    next_word = max(generate_S_unigram, key=generate_S_unigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print( " ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['oleh']
  for i in range(10):
    generate_S_unigram = model.probabilities_for_all_vocab([word[-1]], train, unigram_counts, vocab)
    next_word = max(generate_S_unigram, key=generate_S_unigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print( " ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['pergi']
  for i in range(10):
    generate_S_unigram = model.probabilities_for_all_vocab([word[-1]], train, unigram_counts, vocab)
    next_word = max(generate_S_unigram, key=generate_S_unigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print( " ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['dari']
  for i in range(10):
    generate_S_unigram = model.probabilities_for_all_vocab([word[-1]], train, unigram_counts, vocab)
    next_word = max(generate_S_unigram, key=generate_S_unigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_unigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print( " ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")


  """
  BIGRAM
  """
  print("\nBIGRAM")
  generate_S_bigram = model.probabilities_for_all_vocab(['karena'], unigram_counts, bigram_counts, vocab)
  # print(max(generate_S_bigram, key=generate_S_bigram.get))
  # print(sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[:5])

  word = ['karena']
  for i in range(10):
    generate_S_bigram = model.probabilities_for_all_vocab([word[-1]], unigram_counts, bigram_counts, vocab)
    next_word = max(generate_S_bigram, key=generate_S_bigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print(" ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['Berkas']
  for i in range(10):
    generate_S_bigram = model.probabilities_for_all_vocab([word[-1]], unigram_counts, bigram_counts, vocab)
    next_word = max(generate_S_bigram, key=generate_S_bigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print(" ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['oleh']
  for i in range(10):
    generate_S_bigram = model.probabilities_for_all_vocab([word[-1]], unigram_counts, bigram_counts, vocab)
    next_word = max(generate_S_bigram, key=generate_S_bigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print(" ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['pergi']
  for i in range(10):
    generate_S_bigram = model.probabilities_for_all_vocab([word[-1]], unigram_counts, bigram_counts, vocab)
    next_word = max(generate_S_bigram, key=generate_S_bigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print(" ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  word = ['dari']
  for i in range(10):
    generate_S_bigram = model.probabilities_for_all_vocab([word[-1]], unigram_counts, bigram_counts, vocab)
    next_word = max(generate_S_bigram, key=generate_S_bigram.get)
    if next_word == word[-1]:
      next_word = sorted(generate_S_bigram.items(), key=lambda x:x[1], reverse=True)[1][0]
    word.append(next_word)

  print(" ".join(word))
  perplexity_test_unigram = model.count_perplexity(["<s>"] + word + ["</s>"], unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")
  """
  TRIGRAM
  """
  # generate_S_trigram = model.probabilities_for_all_vocab(['ini', 'karena'], bigram_counts, trigram_counts, vocab)
  # print(max(generate_S_trigram, key=generate_S_trigram.get))
  # print(sorted(generate_S_trigram.items(), key=lambda x:x[1], reverse=True)[:5])

  # word = ['karena', 'itu']
  # for i in range(10):
  #   generate_S_trigram = model.probabilities_for_all_vocab(word[-2:], bigram_counts, trigram_counts, vocab)
  #   next_word = max(generate_S_trigram, key=generate_S_trigram.get)
  #   if next_word == word[-1]:
  #     next_word = sorted(generate_S_trigram.items(), key=lambda x:x[1], reverse=True)[1][0]
  #   word.append(next_word)

  # print("Trigram :", " ".join(word))

  """
  - Di bawah ini merupakan contoh/cara untuk menilai perplexity dari kalimat yang telah Anda generate dari langkah sebelumnya.
  """
  '''
  print('\nPREPLEXITY\n')

  print('\nUNIGRAM')
  print('Kalimat : saya sedang menunggu di peron 5 stasiun tersebut')
  perplexity_test_unigram = model.count_perplexity("<s> saya sedang menunggu di peron 5 stasiun tersebut </s>".split(), train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  print('\nUNIGRAM')
  print('Kalimat : para pekerja terlihat lincah saat membersihkan lokomotif tersebut')
  perplexity_test_unigram = model.count_perplexity("<s> para pekerja terlihat lincah saat membersihkan lokomotif tersebut </s>".split(), train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  print('\nBIGRAM')
  print('Kalimat : pak ustad berceramah di atas mimbar masjid')
  perplexity_test_bigram = model.count_perplexity("<s> <s> pak ustad berceramah di atas mimbar masjid </s>".split(), unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_bigram:.4f}")

  print('\nBIGRAM')
  print('Kalimat : para murid diajarkan budi pekerti di sekolah')
  perplexity_test_bigram = model.count_perplexity("<s> <s> para murid diajarkan budi pekerti di sekolah </s>".split(), unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_bigram:.4f}")
  '''
  
  """
  UNIGRAM
  """
  print('\nUNIGRAM')
  perplexity_test_unigram = model.count_perplexity(flatten_test, train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_unigram:.4f}")

  """
  BIGRAM
  """
  print('\nBIGRAM')
  perplexity_test_bigram = model.count_perplexity(flatten_test, unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_bigram:.4f}")


  """
  TRIGRAM
  """
  print('\nTRIGRAM')
  perplexity_test_trigram = model.count_perplexity(flatten_test, bigram_counts, trigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 2, Perplexity: {perplexity_test_trigram:.4f}")

  """
  TRIGRAM from generated sentence
  """
  perplexity_test_random = model.count_perplexity(['<s>', 'cagar', 'budaya', 'merupakan', 'aset', 'di', 'indonesia', '</s>'], bigram_counts, trigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 2, Perplexity: {perplexity_test_random:.4f}")

  '''
  """
  BIGRAM from generated sentence
  """
  perplexity_test_random = model.count_perplexity("karena itu sendiri yaitu di dalam bahasa Indonesia yang lebih dari".split(), unigram_counts, bigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_random:.4f}")

  """
  UNIGRAM from generated sentence
  """
  perplexity_test_random = model.count_perplexity("karena yang yang yang yang yang yang yang yang yang yang".split(), train, unigram_counts, len(vocab), laplace_number=1.0)
  print(f"n = 1, Perplexity: {perplexity_test_random:.4f}")
  '''
  

if __name__ == "__main__":
  main()
