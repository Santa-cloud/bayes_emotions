import math
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        # Przechowuje parametry modelu
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}

    def fit(self, X, y):
        """
        Trenuje model na podstawie danych treningowych.

        Args:
        X: Tablica numpy z wektorami cech (dane treningowe).
        y: Tablica numpy z etykietami klas odpowiadającymi wektorom cech.
        """
        # Konwersja do tablic numpy (jeśli nie są)
        X = np.array(X)
        y = np.array(y)

        # Wyznacz unikalne klasy
        classes = np.unique(y)

        for cls in classes:
            # Filtruj dane dla danej klasy
            X_cls = X[y == cls]
            # Oblicz średnie i wariancje dla każdej cechy
            self.class_means[cls] = np.mean(X_cls, axis=0)
            self.class_variances[cls] = np.var(X_cls, axis=0) + 1e-6  # Dodaj niewielką stałą
            # Oblicz priorytet klasy jako stosunek liczby próbek w tej klasie do liczby wszystkich próbek
            self.class_priors[cls] = X_cls.shape[0] / X.shape[0]

    def _log_gaussian_probability(self, x, mean, variance):
        """
        Oblicza logarytm prawdopodobieństwa cechy na podstawie rozkładu Gaussa.

        Args:
        x: Wartość cechy.
        mean: Średnia dla danej cechy.
        variance: Wariancja dla danej cechy.

        Returns:
        Logarytm prawdopodobieństwa dla danej cechy.
        """
        exponent = -((x - mean) ** 2) / (2 * variance)
        log_prob = -0.5 * (np.log(2 * np.pi * variance)) + exponent
        return log_prob

    def predict(self, X):
        """
        Dokonuje predykcji dla danych testowych.

        Args:
        X: Tablica numpy z wektorami cech (dane testowe).

        Returns:
        Lista etykiet klas dla danych testowych.
        """
        X = np.array(X)
        predictions = []

        for sample in X:
            class_log_probabilities = {}

            # Oblicz logarytmy prawdopodobieństw a posteriori dla każdej klasy
            for cls in self.class_means:
                mean = self.class_means[cls]
                variance = self.class_variances[cls]
                log_probs = self._log_gaussian_probability(sample, mean, variance)
                log_likelihood = np.sum(log_probs)
                log_prior = math.log(self.class_priors[cls])
                class_log_probabilities[cls] = log_likelihood + log_prior

            # Przypisz klasę o najwyższym logarytmie prawdopodobieństwa
            predicted_class = max(class_log_probabilities, key=class_log_probabilities.get)
            predictions.append(predicted_class)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Ocena klasyfikatora na danych testowych.

        Args:
        X_test: Tablica numpy z wektorami cech (dane testowe).
        y_test: Lista lub tablica numpy z prawdziwymi etykietami klas dla danych testowych.

        Returns:
        Dokładność klasyfikatora oraz raport klasyfikacji.
        """
        predictions = self.predict(X_test)
        correct = sum(1 for true, pred in zip(y_test, predictions) if true == pred)
        accuracy = correct / len(y_test)

        # Generowanie raportu klasyfikacji
        from collections import Counter
        class_counts = Counter(y_test)
        pred_counts = Counter(predictions)
        classes = sorted(class_counts.keys())

        # Nagłówek tabeli
        report = "\nKlasyfikacja na podstawie danych testowych:\n"
        report += "{:<12} {:<10} {:<10} {:<10} {:<10}\n".format('Klasa', 'Precyzja', 'Czułość', 'F1-score', 'Support')
        report += "-" * 54 + "\n"

        for cls in classes:
            tp = sum(1 for true, pred in zip(y_test, predictions) if true == cls and pred == cls)
            fp = sum(1 for true, pred in zip(y_test, predictions) if true != cls and pred == cls)
            fn = sum(1 for true, pred in zip(y_test, predictions) if true == cls and pred != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = class_counts[cls]

            report += "{:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}\n".format(
                cls, precision, recall, f1, support
            )

        report += "-" * 54 + "\n"
        report += "Dokładność modelu: {:.2f}%\n".format(accuracy * 100)

        return accuracy, report

    def score(self, X, y):
        """
        Oblicza dokładność klasyfikatora na danych X i y.

        Args:
        X: Tablica numpy z wektorami cech.
        y: Lista lub tablica numpy z etykietami klas.

        Returns:
        Dokładność klasyfikatora.
        """
        predictions = self.predict(X)
        correct = sum(1 for true, pred in zip(y, predictions) if true == pred)
        return correct / len(y)
