import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

class RAVDESSProblem:
    def __init__(self, data_path, n_mfcc=20, hop_length=512, test_size=0.2):
        """
        Inicjalizuje parametry problemu i ścieżkę do danych.
        
        Args:
        data_path: Ścieżka do folderu z danymi (pliki audio .wav).
        n_mfcc: Liczba współczynników MFCC do wyznaczenia.
        hop_length: Przesunięcie ramek dla funkcji Librosa.
        test_size: Procent danych przeznaczonych na zbiór testowy.
        """
        self.data_path = data_path
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.test_size = test_size
        self.features = []
        self.labels = []

    def extract_features(self):
        """
        Wyodrębnia cechy MFCC i ich wariancje ze wszystkich plików audio w folderze.
        """
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    
                    # Wczytaj plik audio
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Wyodrębnij cechy MFCC i delta MFCC
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
                    delta_mfccs = librosa.feature.delta(mfccs)
                    
                    # Oblicz średnie i wariancje cech
                    mfcc_means = np.mean(mfccs, axis=1)
                    mfcc_vars = np.var(mfccs, axis=1)
                    delta_means = np.mean(delta_mfccs, axis=1)
                    delta_vars = np.var(delta_mfccs, axis=1)
                    
                    # Połącz cechy w jeden wektor
                    feature_vector = np.concatenate([mfcc_means, mfcc_vars, delta_means, delta_vars])
                    
                    # Dodaj wektor cech i etykietę do listy
                    self.features.append(feature_vector)
                    
                    # Wyodrębnij etykietę z nazwy pliku (np. "03-01-06-02-02-01-12.wav")
                    label = int(file.split("-")[2]) - 1  # Etykieta emocji (0-7 dla emocji z RAVDESS)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test
