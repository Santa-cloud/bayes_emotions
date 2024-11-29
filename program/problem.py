# problem.py
import os
import csv
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

class RAVDESSProblem:
    def __init__(self, data_path, n_mfcc=20, hop_length=512, n_fft=2048, test_size=0.2, output_csv="features.csv"):
        """
        Inicjalizuje parametry problemu i ścieżkę do danych.
        
        Args:
        data_path: Ścieżka do folderu z danymi (pliki audio .wav).
        n_mfcc: Liczba współczynników MFCC do wyznaczenia.
        hop_length: Przesunięcie ramek dla funkcji Librosa.
        n_fft: Długość okna FFT.
        test_size: Procent danych przeznaczonych na zbiór testowy.
        output_csv: Nazwa pliku, w którym będą zapisane cechy.
        """
        self.data_path = data_path
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.test_size = test_size
        self.output_csv = output_csv
        self.features = []
        self.labels = []
        self.labels_dict = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

    def extract_features(self):
        """
        Wyodrębnia cechy MFCC i ich wariancje ze wszystkich plików audio w folderze i zapisuje je do pliku CSV.
        """
        # Otwórz plik CSV do zapisu
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Zapisz nagłówki kolumn
            header = ['file_name', 'label'] + [f'mfcc_mean_{i}' for i in range(self.n_mfcc)] + \
                     [f'mfcc_var_{i}' for i in range(self.n_mfcc)] + \
                     [f'delta_mean_{i}' for i in range(self.n_mfcc)] + \
                     [f'delta_var_{i}' for i in range(self.n_mfcc)]
            writer.writerow(header)
            
            for root, dirs, files in os.walk(self.data_path):
                # Pomijamy folder 'venv'
                dirs[:] = [d for d in dirs if d != 'venv']
                for file in files:
                    if file.endswith(".wav"):
                        file_path = os.path.join(root, file)
                        print(f'Wyodrębnianie cech z pliku: {file_path}')
                        
                        # Wyodrębnij etykietę z nazwy pliku
                        try:
                            parts = file.split("-")
                            if len(parts) < 3:
                                print(f"Nazwa pliku {file} nie jest w oczekiwanym formacie. Pomijanie pliku.")
                                continue
                            label_code = parts[2]
                            label = self.labels_dict[label_code]
                        except (IndexError, KeyError) as e:
                            print(f"Błąd podczas wyodrębniania etykiety z pliku {file}: {e}")
                            continue  # Pomija ten plik i przechodzi do następnego
                        
                        # Wczytaj plik audio i wyodrębnij cechy
                        try:
                            y, sr = librosa.load(file_path, sr=None)
                            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length, n_fft=self.n_fft)
                            delta_mfccs = librosa.feature.delta(mfccs)
                            mfcc_means = np.mean(mfccs, axis=1)
                            mfcc_vars = np.var(mfccs, axis=1)
                            delta_means = np.mean(delta_mfccs, axis=1)
                            delta_vars = np.var(delta_mfccs, axis=1)
                            feature_vector = np.concatenate([mfcc_means, mfcc_vars, delta_means, delta_vars])
                        except Exception as e:
                            print(f"Błąd podczas przetwarzania pliku {file}: {e}")
                            continue  # Pomija ten plik i przechodzi do następnego
                        
                        # Dodaj dane do pliku CSV
                        row = [file, label] + feature_vector.tolist()
                        writer.writerow(row)
                        
                        # Dodaj cechy i etykiety do wewnętrznych list
                        self.features.append(feature_vector)
                        self.labels.append(label)

    def prepare_data(self):
        """
        Przygotowuje dane do treningu i testowania klasyfikatora.
        
        Returns:
        X_train, X_test, y_train, y_test: Dane podzielone na zbiory treningowy i testowy.
        """
        self.extract_features()
        
        # Zamiana list na tablice numpy
        X = np.array(self.features)
        y = np.array(self.labels)
        
        # Podział na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=12)
        return X_train, X_test, y_train, y_test
