from problem import RAVDESSProblem
from method import NaiveBayesClassifier
from sklearn.naive_bayes import GaussianNB
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ProgramState:
    """
    Klasa przechowująca parametry i stan programu.
    """
    def __init__(self):
        # Domyślne parametry
        self.data_path = r"C:\politechnika OKNO\R4_1PS\Metody Sztuczej Inteligencji\projekt\data"
        self.output_csv = "features.csv"
        self.n_mfcc = 20
        self.hop_length = 512
        self.n_fft = 2048
        self.test_size = 0.2

        # Inicjalizacja problemu i klasyfikatorów
        self.problem = None
        self.custom_classifier = NaiveBayesClassifier()
        self.sklearn_classifier = GaussianNB()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

def display_menu():
    """
    Wyświetla menu użytkownika.
    """
    print("\nMenu:")
    print("1. Ustaw parametry przetwarzania")
    print("2. Wyodrębnij cechy i zapisz do pliku CSV")
    print("3. Trenuj model z danych CSV")
    print("4. Testuj klasyfikator")
    print("5. Wyjdź z programu")
    print("-" * 30)

def set_parameters(state):
    """
    Ustawia parametry przetwarzania na podstawie danych od użytkownika.
    """
    print("\nUstawianie parametrów przetwarzania:")
    state.data_path = input(f"Podaj ścieżkę do danych (domyślnie: {state.data_path}): ") or state.data_path
    state.output_csv = input(f"Podaj nazwę pliku CSV (domyślnie: {state.output_csv}): ") or state.output_csv

    try:
        n_mfcc_input = input(f"Podaj liczbę cech MFCC (12-30, domyślnie: {state.n_mfcc}): ") or state.n_mfcc
        state.n_mfcc = int(n_mfcc_input)
        if not 12 <= state.n_mfcc <= 30:
            raise ValueError("Liczba cech MFCC musi być w zakresie 12-30.")
    except ValueError as e:
        print(f"Błąd: {e}. Powrót do poprzedniej wartości.")

    try:
        hop_length_input = input(f"Podaj stopień przeplotu ramek (domyślnie: {state.hop_length}): ") or state.hop_length
        state.hop_length = int(hop_length_input)
        if state.hop_length <= 0:
            raise ValueError("Stopień przeplotu ramek musi być większy od 0.")
    except ValueError as e:
        print(f"Błąd: {e}. Powrót do poprzedniej wartości.")

    try:
        n_fft_input = input(f"Podaj długość okna ramki (n_fft) (domyślnie: {state.n_fft}): ") or state.n_fft
        state.n_fft = int(n_fft_input)
        if state.n_fft <= 0:
            raise ValueError("Długość okna ramki musi być większa od 0.")
    except ValueError as e:
        print(f"Błąd: {e}. Powrót do poprzedniej wartości.")

    try:
        test_size_input = input(f"Podaj udział danych testowych w zbiorze (0.1-0.5, domyślnie: {state.test_size}): ") or state.test_size
        state.test_size = float(test_size_input)
        if not 0.1 <= state.test_size <= 0.5:
            raise ValueError("Udział danych testowych musi być w zakresie 0.1-0.5.")
    except ValueError as e:
        print(f"Błąd: {e}. Powrót do poprzedniej wartości.")

    print("\nParametry ustawione pomyślnie.")

def extract_and_save_features(state):
    """
    Wyodrębnia cechy z plików audio i zapisuje je do pliku CSV.
    """
    print("\nEkstrakcja cech i zapis do pliku CSV:")
    if not state.data_path:
        print("Błąd: Nie ustawiono ścieżki do danych. Ustaw parametry w menu.")
        return

    # Inicjalizacja problemu
    state.problem = RAVDESSProblem(
        data_path=state.data_path,
        n_mfcc=state.n_mfcc,
        hop_length=state.hop_length,
        n_fft=state.n_fft,
        test_size=state.test_size,
        output_csv=state.output_csv
    )
    state.problem.extract_features()
    print(f"Cechy zostały zapisane do pliku {state.output_csv}.")

def train_model(state):
    """
    Trenuje model na podstawie danych z pliku CSV.
    """
    print("\nTrenowanie modelu z danych CSV:")
    if not os.path.exists(state.output_csv):
        print(f"Błąd: Plik {state.output_csv} nie istnieje. Wykonaj najpierw ekstrakcję cech.")
        return

    # Wczytaj dane z pliku CSV
    data = pd.read_csv(state.output_csv)
    X = data.drop(columns=['file_name', 'label']).values
    y = data['label'].values

    # Podział na zbiory treningowy i testowy
    state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
        X, y, test_size=state.test_size, random_state=12
    )

    print("\nWybierz klasyfikator:")
    print("1. Własna implementacja Naive Bayes")
    print("2. Gaussian Naive Bayes z sklearn")
    classifier_choice = input("Wybierz opcję: ")

    if classifier_choice == "1":
        state.custom_classifier.fit(state.X_train, state.y_train)
        print("Model został wytrenowany przy użyciu własnej implementacji Naive Bayes.")
    elif classifier_choice == "2":
        state.sklearn_classifier.fit(state.X_train, state.y_train)
        print("Model został wytrenowany przy użyciu Gaussian Naive Bayes z sklearn.")
    else:
        print("Nieprawidłowy wybór. Powrót do menu.")

def test_classifier(state):
    """
    Testuje wytrenowany klasyfikator na zbiorze testowym.
    """
    print("\nTestowanie klasyfikatora:")
    if state.X_test is None or state.y_test is None:
        print("Błąd: Nie przeprowadzono procesu trenowania. Najpierw wykonaj trenowanie.")
        return

    print("\nWybierz klasyfikator do testowania:")
    print("1. Własna implementacja Naive Bayes")
    print("2. Gaussian Naive Bayes z sklearn")
    classifier_choice = input("Wybierz opcję: ")

    if classifier_choice == "1":
        # Testowanie własnej implementacji
        accuracy, classification_report_custom = state.custom_classifier.evaluate(state.X_test, state.y_test)
        print(classification_report_custom)
    elif classifier_choice == "2":
        # Testowanie GaussianNB z sklearn
        predictions = state.sklearn_classifier.predict(state.X_test)
        accuracy = state.sklearn_classifier.score(state.X_test, state.y_test)
        print(f"\nDokładność: {accuracy * 100:.2f}%")
        print("\nRaport klasyfikacji:")
        print(classification_report(state.y_test, predictions))
    else:
        print("Nieprawidłowy wybór. Powrót do menu.")

def main():
    """
    Główna funkcja sterująca programem.
    """
    # Inicjalizacja stanu programu
    state = ProgramState()

    while True:
        display_menu()
        choice = input("Wybierz opcję: ")

        if choice == "1":
            set_parameters(state)
        elif choice == "2":
            extract_and_save_features(state)
        elif choice == "3":
            train_model(state)
        elif choice == "4":
            test_classifier(state)
        elif choice == "5":
            print("Wyjście z programu. Do widzenia!")
            break
        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    main()
