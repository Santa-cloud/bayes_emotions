from problem import RAVDESSProblem
from method import NaiveBayesClassifier

def display_menu():
    """
    Wyświetla menu użytkownika.
    """
    print("\nMenu:")
    print("1. Ustaw parametry przetwarzania")
    print("2. Uruchom proces uczenia")
    print("3. Testuj klasyfikator")
    print("4. Wyjdź z programu")
    print("-" * 30)

def main():
    """
    Główna funkcja sterująca programem.
    """
    # Domyślne parametry
    data_path = r"C:\politechnika OKNO\R4_1PS\Metody Sztuczej Inteligencji\projekt\data"
    n_mfcc = 20
    hop_length = 512
    test_size = 0.2

    # Inicjalizacja problemu i klasyfikatora
    problem = None
    classifier = NaiveBayesClassifier()

    while True:
        display_menu()
        choice = input("Wybierz opcję: ")

        if choice == "1":
            print("\nUstawianie parametrów przetwarzania:")
            data_path = input(f"Podaj ścieżkę do danych (domyślnie: {data_path}): ") or data_path
            try:
                n_mfcc = int(input(f"Podaj liczbę cech MFCC (12-30, domyślnie: {n_mfcc}): ") or n_mfcc)
                if not 12 <= n_mfcc <= 30:
                    raise ValueError("Liczba cech MFCC musi być w zakresie 12-30.")
            except ValueError as e:
                print(f"Błąd: {e}. Powrót do domyślnych wartości.")
                n_mfcc = 20

            try:
                hop_length = int(input(f"Podaj stopień przeplotu ramek (domyślnie: {hop_length}): ") or hop_length)
                if hop_length <= 0:
                    raise ValueError("Stopień przeplotu ramek musi być większy od 0.")
            except ValueError as e:
                print(f"Błąd: {e}. Powrót do domyślnych wartości.")
                hop_length = 512

            try:
                test_size = float(input(f"Podaj procentowy podział danych na testowe (0.1-0.5, domyślnie: {test_size}): ") or test_size)
                if not 0.1 <= test_size <= 0.5:
                    raise ValueError("Procentowy podział danych musi być w zakresie 0.1-0.5.")
            except ValueError as e:
                print(f"Błąd: {e}. Powrót do domyślnych wartości.")
                test_size = 0.2

            print("\nParametry ustawione pomyślnie.")

        elif choice == "2":
            print("\nUruchamianie procesu uczenia:")
            if not data_path:
                print("Błąd: Nie ustawiono ścieżki do danych. Ustaw parametry w menu.")
                continue

            # Inicjalizacja problemu
            problem = RAVDESSProblem(data_path, n_mfcc=n_mfcc, hop_length=hop_length, test_size=test_size)
            X_train, X_test, y_train, y_test = problem.prepare_data()

            # Trening klasyfikatora
            classifier.fit(X_train, y_train)
            print("Klasyfikator został wytrenowany.")

        elif choice == "3":
            print("\nTestowanie klasyfikatora:")
            if problem is None:
                print("Błąd: Nie przeprowadzono procesu uczenia. Najpierw uruchom proces uczenia.")
                continue

            # Testowanie klasyfikatora
            X_train, X_test, y_train, y_test = problem.prepare_data()
            accuracy, classification_report = classifier.evaluate(X_test, y_test)
            print(f"\nDokładność: {accuracy * 100:.2f}%")
            print("Raport klasyfikacji:")
            print(classification_report)

        elif choice == "4":
            print("Wyjście z programu. Do widzenia!")
            break

        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    main()
