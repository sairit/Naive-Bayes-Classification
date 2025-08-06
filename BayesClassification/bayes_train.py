"""
@Author: Sai Yadavalli
Version: 2.1
"""


import json
import os

class TextProcessor:
    def __init__(self, training_file):
        """
        Initializes the TextProcessor class.

        Args:
            training_file (str): Path to the training file.
        """
        self.training_file = training_file
        self.stop_words_file = os.path.join(os.path.dirname(training_file), "StopWords.txt")
        self.stop_words = set()
        self.word_counts = {}  # [ham_count, spam_count]
        self.word_probabilities = {}
        self.total_spam = 0
        self.total_ham = 0
        self.k = 1  # smoothing factor

    def load_stop_words(self):
        """
        Loads stop words from the stop words file.
        """
        with open(self.stop_words_file, "r", encoding="unicode-escape") as f:
            self.stop_words = set(f.read().splitlines())

    @staticmethod
    def cleantext(text):
        """
        Cleans the input text by converting to lowercase, removing punctuation, and stripping whitespace.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        text = text.lower()
        text = text.strip()
        for letters in text:
            if letters in """[]!.,"-!—@;':#$%^&*()+/?""":
                text = text.replace(letters, " ")
        return text

    def process_training_data(self):
        """
        Processes the training data to calculate word counts for ham and spam emails.
        """
        with open(self.training_file, "r", encoding="unicode-escape") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label = line[0]
                subject = line[2:]
                words = self.cleantext(subject).split()
                filtered_words = []
                for word in words:
                    if word not in self.stop_words:
                        filtered_words.append(word)

                if label == "1":
                    self.total_spam += 1
                elif label == "0":
                    self.total_ham += 1

                for word in filtered_words:
                    if word not in self.word_counts:
                        self.word_counts[word] = [0, 0]
                    if label == "1":
                        self.word_counts[word][1] += 1
                    elif label == "0":
                        self.word_counts[word][0] += 1

    def calculate_probabilities(self):
        """
        Calculates the probabilities of each word being in ham or spam emails using Laplace smoothing.
        """
        for word, (ham_count, spam_count) in self.word_counts.items():
            ham_prob = (ham_count + self.k) / (2 * self.k + self.total_ham)
            spam_prob = (spam_count + self.k) / (2 * self.k + self.total_spam)
            self.word_probabilities[word] = [ham_prob, spam_prob]

    def save_processed_data(self):
        """
        Saves the processed data (word probabilities and total counts) to files.
        """
        os.makedirs("BayesClassification/train", exist_ok=True)

        self.calculate_probabilities()

        with open("BayesClassification/train/Dictionary.txt", "w") as f:
            json.dump(self.word_probabilities, f, indent=4)

        with open("BayesClassification/train/HSCount.txt", "w") as f:
            f.write(f"{self.total_ham} {self.total_spam}")



# MAIN
# Step 1: Process and save the data
training_file = input("Enter the path to the training file: ")

processor = TextProcessor(training_file)
processor.load_stop_words()
processor.process_training_data()
processor.save_processed_data()