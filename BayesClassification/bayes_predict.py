"""
@Author: Sai Yadavalli (modified)
Version: 1.4
"""

import json
import matplotlib.pyplot as plt
import numpy as np


class BayesClassification:
    def __init__(self, hscount_file, dictionary_file, test_file):
        """
        Initializes the BayesClassification class with file paths.

        Args:
            hscount_file (str): Path to the HSCount file.
            dictionary_file (str): Path to the dictionary file.
            test_file (str): Path to the test file.
        """
        self.hscount_file = hscount_file
        self.dictionary_file = dictionary_file
        self.test_file = test_file
        self.total_ham = 0
        self.total_spam = 0
        self.word_probabilities = {}

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
        for letters in """[]!.,"-!—@;':#$%^&*()+/?""":
            text = text.replace(letters, " ")
        return text

    def load_model(self):
        """
        Loads the model data from the HSCount and dictionary files.
        """
        with open(self.hscount_file, "r", encoding="utf-8") as f:
            counts = f.read().split()
            self.total_ham = int(counts[0])
            self.total_spam = int(counts[1])

        with open(self.dictionary_file, "r", encoding="utf-8") as f:
            self.word_probabilities = json.load(f)

    def test_cutoff(self, cutoff):
        """
        Tests the model using the test file and calculates performance metrics for a given cutoff.

        Args:
            cutoff (float): The cutoff value for classifying spam vs ham.

        Returns:
            tuple: A tuple containing TP, FP, TN, FN, precision, accuracy, recall, and F1 score.
        """
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with open(self.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                label = line[0]
                subject = line[2:]
                words = self.cleantext(subject).split()

                spam_probability = 1
                ham_probability = 1

                for word in self.word_probabilities:
                    if word in words:
                        spam_probability *= self.word_probabilities[word][1]
                        ham_probability *= self.word_probabilities[word][0]
                    else:
                        spam_probability *= (1 - self.word_probabilities[word][1])
                        ham_probability *= (1 - self.word_probabilities[word][0])

                if spam_probability > cutoff * ham_probability:
                    prediction = "1"
                else:
                    prediction = "0"

                if prediction == label:
                    if prediction == "1":
                        TP += 1
                    else:
                        TN += 1
                else:
                    if prediction == "1":
                        FP += 1
                    else:
                        FN += 1

        total = TP + FP + TN + FN
        accuracy = (TP + TN) / total if total else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        return TP, FP, TN, FN, precision, accuracy, recall, f1

    def evaluate_cutoffs(self):
        """
        Evaluates the model for multiple cutoff values and generates metrics for each.

        Returns:
            list: A list of dictionaries containing metrics for each cutoff.
        """
        cutoffs = []
        for x in np.arange(0, 0.95, 0.05):
            cutoffs.append(round(x, 2))

        results = []
        for cutoff in cutoffs:
            TP, FP, TN, FN, precision, accuracy, recall, f1 = self.test_cutoff(cutoff)

            tp_rate = TP / self.total_spam if self.total_spam else 0
            fp_rate = FP / self.total_ham if self.total_ham else 0
            tn_rate = TN / self.total_ham if self.total_ham else 0
            fn_rate = FN / self.total_spam if self.total_spam else 0

            result = {
                "cutoff": cutoff,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                "tn_rate": tn_rate,
                "fn_rate": fn_rate,
            }
            results.append(result)

            print(f"Cutoff: {cutoff:.2f}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        self.plot_metrics(results)
        return results

    def plot_metrics(self, results):
        """
        Generates plots for the metrics based on the results.

        Args:
            results (list): A list of dictionaries containing metrics for each cutoff.
        """
        cutoffs = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        tp_rates = []
        fp_rates = []
        tn_rates = []
        fn_rates = []

        for result in results:
            cutoffs.append(result["cutoff"])
            accuracies.append(result["accuracy"])
            precisions.append(result["precision"])
            recalls.append(result["recall"])
            f1_scores.append(result["f1_score"])
            tp_rates.append(result["tp_rate"])
            fp_rates.append(result["fp_rate"])
            tn_rates.append(result["tn_rate"])
            fn_rates.append(result["fn_rate"])

        # Plot 1: Cutoff vs Accuracy, Precision, Recall, F1 Score
        plt.figure(figsize=(10, 6))
        plt.plot(cutoffs, accuracies, label="Accuracy")
        plt.plot(cutoffs, precisions, label="Precision")
        plt.plot(cutoffs, recalls, label="Recall")
        plt.plot(cutoffs, f1_scores, label="F1 Score")
        plt.xlabel("Cutoff")
        plt.ylabel("Metrics")
        plt.title("Cutoff vs Performance Metrics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cutoff_vs_metrics.png")
        plt.show()

        # Plot 2: Cutoff vs TP/Spam, FP/Ham, TN/Ham, FN/Spam
        plt.figure(figsize=(10, 6))
        plt.plot(cutoffs, tp_rates, label="TP/Spam")
        plt.plot(cutoffs, fp_rates, label="FP/Ham")
        plt.plot(cutoffs, tn_rates, label="TN/Ham")
        plt.plot(cutoffs, fn_rates, label="FN/Spam")
        plt.xlabel("Cutoff")
        plt.ylabel("Rates")
        plt.title("Cutoff vs Classification Rates")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cutoff_vs_rates.png")
        plt.show()


#MAIN
# Step 2: Test and evaluate the Bayes classifier
print()
print("HSCount.txt and Dictionary.txt should be located in the train directory, if Sai_Yadavalli_Train.py was ran")
print()

hscount_path = input("Enter the path to HSCount.txt: ")
dictionary_path = input("Enter the path to Dictionary.txt: ")
test_file_path = input("Enter the path to the test file: ")

classifier = BayesClassification(hscount_path, dictionary_path, test_file_path)
classifier.load_model()
#classifier.evaluate_cutoffs()

# Allow the user to test specific cutoff values interactively
testing = True
while testing:
    try:
        cutoff = float(input("Enter your test cutoff (e.g., 0.1, 0.2, etc.): "))
        if 0 <= cutoff <= 1:
            TP, FP, TN, FN, precision, accuracy, recall, f1 = classifier.test_cutoff(cutoff)
            print(f"\nResults for Cutoff: {cutoff}")
            print(f"True Positives (TP): {TP}")
            print(f"False Positives (FP): {FP}")
            print(f"True Negatives (TN): {TN}")
            print(f"False Negatives (FN): {FN}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}\n")
        else:
            print("Please enter a cutoff value between 0 and 1.")
    except ValueError:
        print("Invalid input. Please enter a numeric cutoff value.")

    another = input("Do you want to test another cutoff? (yes/no): ").strip().lower()
    if another != "yes":
        testing = False
