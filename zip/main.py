import ngram_lm
import preprocess_data

def main():
  preprocess_data.main()
  ngram_lm.main()

if __name__ == "__main__":
  main()
