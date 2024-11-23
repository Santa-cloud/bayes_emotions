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
        X: Lista wektorów cech (dane treningowe).
        y: Lista etykiet klas odpowiadających wektorom cech.
        """
        # Wyznacz unikalne klasy
        classes = set(y)
        
        for cls in classes:
            # Filtruj dane dla danej klasy
            X_cls = [X[i] for i in range(len(y)) if y[i] == cls]
            X_cls = np.array(X_cls)
            
            # Oblicz średnie i wariancje dla każdej cechy
            self.class_means[cls] = np.mean(X_cls, axis=0)
            self.class_variances[cls] = np.var(X_cls, axis=0)
            
            # Oblicz priorytet klasy jako stosunek liczby próbek w tej klasie do liczby wszystkich próbek
            self.class_priors[cls] = len(X_cls) / len(X)

    def _gaussian_probability(self, x, mean, variance):
        """
        Oblicza prawdopodobieństwo cechy na podstawie rozkładu Gaussa.
        
        Args:
        x: Wartość cechy.
        mean: Średnia dla danej cechy.
        variance: Wariancja dla danej cechy.
        
        Returns:
        Prawdopodobieństwo dla danej cechy.
        """
        if variance == 0:
            # Unikaj dzielenia przez 0 w przypadku braku wariancji
            variance = 1e-6
        exponent = math.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / math.sqrt(2 * math.pi * variance)) * exponent

    def predict(self, X):
        """
        Dokonuje predykcji dla danych testowych.
        
        Args:
        X: Lista wektorów cech (dane testowe).
        
        Returns:
        Lista etykiet klas dla danych testowych.
        """
        predictions = []
        
        for sample in X:
            class_probabilities = {}
            
            # Oblicz prawdopodobieństwa a posteriori dla każdej klasy
            for cls in self.class_means:
                likelihood = 1
                for i in range(len(sample)):
                    likelihood *= self._gaussian_probability(
                        sample[i], 
                        self.class_means[cls][i], 
                        self.class_variances[cls][i]
                    )
                class_probabilities[cls] = likelihood * self.class_priors[cls]
            
            # Przypisz klasę o najwyższym prawdopodobieństwie
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        
        return predictions

    # def evaluate(self, X, y):
    #     """
    #     Ocena dokładności modelu.
        
    #     Args:
    #     X: Lista wektorów cech (dane testowe).
    #     y: Lista prawdziwych etykiet klas dla danych testowych.
        
    #     Returns:
    #     Dokładność klasyfikatora.
    #     """
    #     predictions = self.predict(X)
    #     correct = sum([1 for i in range(len(y)) if predictions[i] == y[i]])
    #     return correct / len(y)


    def evaluate(self, X_test, y_test):
        """
        Ocena klasyfikatora na danych testowych.
        """
        predictions = self.predict(X_test)
        correct = sum(1 for true, pred in zip(y_test, predictions) if true == pred)
        accuracy = correct / len(y_test)

        # Generowanie raportu klasyfikacji
        from collections import Counter
        report = "Klasyfikacja na podstawie danych testowych:\n"
        class_counts = Counter(y_test)
        for cls in class_counts:
            cls_correct = sum(1 for true, pred in zip(y_test, predictions) if true == pred and true == cls)
            cls_total = class_counts[cls]
            precision = cls_correct / predictions.count(cls) if predictions.count(cls) > 0 else 0
            recall = cls_correct / cls_total
            report += f"Klasa {cls}: Precyzja: {precision:.2f}, Czułość: {recall:.2f}\n"

        return accuracy, report