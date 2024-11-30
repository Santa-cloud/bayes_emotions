# method.py

import numpy as np
from sklearn.metrics import classification_report

class NaiveBayesClassifier:
    def __init__(self):
        """
        Inicjalizuje strukturę klasyfikatora Naive Bayes.
        """
        self.classes = None          # Lista unikalnych klas
        self.class_priors = {}       # Prawdopodobieństwa a priori P(Ω_k)
        self.mean = {}               # Średnie cech dla każdej klasy μ_k,i
        self.var = {}                # Wariancje cech dla każdej klasy σ_k,i^2

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
        self.classes = np.unique(y)
        n_features = X.shape[1]

        for cls in self.classes:
            # Filtruj dane dla danej klasy
            X_c = X[y == cls]
            # Oblicz prawdopodobieństwo a priori P(Ω_k)
            self.class_priors[cls] = X_c.shape[0] / X.shape[0]
            # Oblicz średnie μ_k,i
            self.mean[cls] = np.mean(X_c, axis=0)
            # Oblicz wariancje σ_k,i^2 (dodajemy niewielką stałą, aby uniknąć dzielenia przez zero)
            self.var[cls] = np.var(X_c, axis=0) + 1e-6

    def _log_gaussian_density(self, x, cls):
        """
        Oblicza logarytm gęstości prawdopodobieństwa dla rozkładu Gaussa.

        Args:
        x: Wektor cech pojedynczej próbki.
        cls: Klasa, dla której obliczamy gęstość.

        Returns:
        Wektor logarytmów gęstości dla każdej cechy.
        """
        mean = self.mean[cls]
        var = self.var[cls]
        # Obliczanie składnika wykładniczego
        numerator = -0.5 * ((x - mean) ** 2) / var
        # Obliczanie składnika normalizacyjnego
        denominator = -0.5 * np.log(2 * np.pi * var)
        return numerator + denominator

    def _predict_single(self, x):
        """
        Przewiduje klasę dla pojedynczej próbki.

        Args:
        x: Wektor cech pojedynczej próbki.

        Returns:
        Przewidywana klasa dla próbki x.
        """
        posteriors = []

        for cls in self.classes:
            # Obliczanie logarytmu prawdopodobieństwa a priori log(P(Ω_k))
            prior = np.log(self.class_priors[cls])
            # Obliczanie logarytmu prawdopodobieństwa warunkowego sum_{i} log(P(c_i | Ω_k))
            conditional = np.sum(self._log_gaussian_density(x, cls))
            # Łączne logarytmy: log(P(Ω_k)) + sum_{i} log(P(c_i | Ω_k))
            posterior = prior + conditional
            posteriors.append(posterior)

        # Wybieramy klasę z największym logarytmem prawdopodobieństwa a posteriori
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Dokonuje predykcji dla danych testowych.

        Args:
        X: Tablica numpy z wektorami cech (dane testowe).

        Returns:
        Tablica etykiet klas dla danych testowych.
        """
        X = np.array(X)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def evaluate(self, X_test, y_test):
        """
        Ocena klasyfikatora na danych testowych.

        Args:
        X_test: Tablica numpy z wektorami cech (dane testowe).
        y_test: Tablica numpy z prawdziwymi etykietami klas dla danych testowych.

        Returns:
        Dokładność klasyfikatora oraz raport klasyfikacji.
        """
        y_pred = self.predict(X_test)
        # Obliczanie dokładności
        accuracy = np.mean(y_pred == y_test)
        # Generowanie raportu klasyfikacji
        report = classification_report(y_test, y_pred)
        return accuracy, report

