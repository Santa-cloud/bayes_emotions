# parameter_tuning.py

import os
import csv
import numpy as np
import pandas as pd
from problem import RAVDESSProblem
from method import NaiveBayesClassifier
from sklearn.model_selection import train_test_split

def parameter_tuning():
    """
    Manipuluje parametrami wejściowymi, trenuje model i zapisuje wyniki do pliku CSV.
    """
    # Ścieżka do danych
    data_path = r"C:\politechnika OKNO\R4_1PS\Metody Sztuczej Inteligencji\projekt\data"
    if not os.path.exists(data_path):
        print(f"Błąd: Ścieżka {data_path} nie istnieje.")
        return

    # Ustawienie zakresów parametrów do testowania
    n_mfcc_values = [12, 16, 20, 24, 28, 30]
    hop_length_values = [256, 512, 1024]
    n_fft_values = [512, 1024, 2048]
    test_size_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Plik CSV do zapisu wyników
    results_csv = "parameter_tuning_results.csv"

    # Otwórz plik CSV do zapisu wyników
    with open(results_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Zapisz nagłówki kolumn
        header = ['n_mfcc', 'hop_length', 'n_fft', 'test_size', 'accuracy']
        writer.writerow(header)

        # Iteracja po wszystkich kombinacjach parametrów
        total_iterations = len(n_mfcc_values) * len(hop_length_values) * len(n_fft_values) * len(test_size_values)
        iteration = 1

        for n_mfcc in n_mfcc_values:
            for hop_length in hop_length_values:
                for n_fft in n_fft_values:
                    for test_size in test_size_values:
                        print(f"\nIteracja {iteration}/{total_iterations}")
                        print(f"Parametry: n_mfcc={n_mfcc}, hop_length={hop_length}, n_fft={n_fft}, test_size={test_size}")

                        # Nazwa pliku CSV dla cech (unikalna dla każdej kombinacji parametrów)
                        output_csv = f"features_mfcc{n_mfcc}_hop{hop_length}_fft{n_fft}.csv"

                        # Inicjalizacja problemu
                        problem = RAVDESSProblem(
                            data_path=data_path,
                            n_mfcc=n_mfcc,
                            hop_length=hop_length,
                            n_fft=n_fft,
                            test_size=test_size,
                            output_csv=output_csv
                        )

                        # Sprawdź, czy plik cech już istnieje, aby uniknąć ponownej ekstrakcji
                        if not os.path.exists(output_csv):
                            # Ekstrakcja cech
                            print("Wyodrębnianie cech...")
                            problem.extract_features()
                        else:
                            print("Plik cech już istnieje. Pomijanie ekstrakcji.")

                        # Wczytaj dane z pliku CSV
                        data = pd.read_csv(output_csv)
                        X = data.drop(columns=['file_name', 'label']).values
                        y = data['label'].values

                        # Podział na zbiory treningowy i testowy
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=12
                        )

                        # Inicjalizacja i trening własnego klasyfikatora
                        classifier = NaiveBayesClassifier()
                        classifier.fit(X_train, y_train)

                        # Ewaluacja klasyfikatora
                        accuracy, _ = classifier.evaluate(X_test, y_test)
                        accuracy_percentage = accuracy * 100
                        print(f"Dokładność: {accuracy_percentage:.2f}%")

                        # Zapisz wyniki do pliku CSV
                        row = [n_mfcc, hop_length, n_fft, test_size, accuracy_percentage]
                        writer.writerow(row)

                        iteration += 1

    print(f"\nZakończono tunowanie parametrów. Wyniki zapisano w pliku {results_csv}.")

if __name__ == "__main__":
    parameter_tuning()