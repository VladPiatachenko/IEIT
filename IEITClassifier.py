import numpy as np

class IEITClassifier:
    def __init__(self):
        self.train_features = None
        self.class_means = None
        self.distance_matrix = None
        self.closest_pairs = None
        self.optimal_radii = {}  # Оптимальні радіуси для кожного класу

    def fit(self, train_features, train_labels):
        """
        Формування правил класифікатора:
        - Обчислення бінарних середніх векторів для кожного класу
        - Визначення найближчого класу для кожного класу
        - Обчислення параметрів достовірності та помилки для кожного радіуса
        """
        self.train_features = train_features

        unique_classes = np.unique(train_labels)

        # Обчислення бінарних середніх векторів для кожного класу
        self.class_means = []
        for class_label in unique_classes:
            class_features = train_features[train_labels == class_label]
            class_mean = np.mean(class_features, axis=0)
            binary_class_mean = (class_mean >= 0.5).astype(int)
            self.class_means.append(binary_class_mean)
        self.class_means = np.array(self.class_means)

        # Розрахунок матриці відстаней
        self.distance_matrix = self.calculate_distance_matrix(self.class_means)

        # Визначення найближчих пар класів
        self.closest_pairs = self.find_all_closest_pairs(self.distance_matrix)

        # Обчислення коефіцієнтів
        self.compute_reliability_and_error()

    @staticmethod
    def hamming_distance(vector1, vector2):
        """Розраховує гаммінгову відстань між двома бінарними векторами"""
        return np.sum(vector1 != vector2)

    def calculate_distance_matrix(self, class_means):
        """Створює матрицю гаммінгових відстаней між класами"""
        num_classes = class_means.shape[0]
        distance_matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distance = self.hamming_distance(class_means[i], class_means[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix

    def find_all_closest_pairs(self, distance_matrix):
        """Для кожного класу знаходить найближчий до нього клас"""
        num_classes = distance_matrix.shape[0]
        closest_pairs = {}
        for i in range(num_classes):
            distances = distance_matrix[i].copy()
            distances[i] = np.inf  # Ігноруємо відстань до самого себе
            closest_class = np.argmin(distances)
            closest_pairs[i] = closest_class
        return closest_pairs

    def compute_Jm(self, K1, K3):
        """
        Обчислення Jm згідно з формулою.
        """
        n = 1  # Припущення рівноімовірності
        r = 3
        epsilon = 10**(-r)  # Дуже мале число для стабілізації

        numerator = (2 * n + epsilon - (1 - K1 + K3))
        denominator = (1 - K1 + K3 + epsilon)

        if denominator == 0:
            return float('-inf')  # Запобігання діленню на 0

        log_term = np.log2(numerator / denominator)

        Jm = (1 / n) * log_term * (n - (1 - K1 + K3))

        return Jm

    def compute_reliability_and_error(self):
        """
        Виконує цикл по радіусу:
        - Обчислення відстаней для кожного класу
        - Розрахунок коефіцієнта на кожному радіусі
        - Збереження оптимального радіуса
        """
        num_features = self.train_features.shape[1]  # Число ознак у векторі
        unique_classes = range(len(self.class_means))

        print("\n--- Розрахунок параметрів на кожному радіусі ---")

        for class_label in unique_classes:
            neighbor_class_label = self.closest_pairs[class_label]
            mean_vector = self.class_means[class_label]

            class_vectors = self.train_features[class_label == train_labels]
            neighbor_vectors = self.train_features[neighbor_class_label == train_labels]

            optimal_radius = None

            print(f"\nКлас {class_label} (найближчий клас {neighbor_class_label})")

            for radius in range(1, num_features + 1):  # Цикл по радіусу
                distances_self = np.array([self.hamming_distance(mean_vector, vec) for vec in class_vectors])
                distances_neighbor = np.array([self.hamming_distance(mean_vector, vec) for vec in neighbor_vectors])

                k1 = np.sum(distances_self <= radius)
                k2 = np.sum(distances_neighbor <= radius)

                reliability = k1 / len(class_vectors)
                beta_error = k2 / len(neighbor_vectors)

                is_optimal = reliability > 0.5 and beta_error < 0.5

                if is_optimal:
                    optimal_radius = radius

                kfe = self.compute_Jm(reliability, beta_error)

                print(f"Радіус {radius}: k1={k1}, k2={k2}, достовірність={reliability:.3f}, помилка={beta_error:.3f}, Jm={kfe:.3f}, Оптимально: {is_optimal}")

            self.optimal_radii[class_label] = optimal_radius

        print("\n--- Оптимальні радіуси для кожного класу ---")
        for class_label, radius in self.optimal_radii.items():
            print(f"Клас {class_label}: Оптимальний радіус = {radius}")

    def predict(self, test_vector):
        """
        Передбачення одного вектора.

        Parameters:
        test_vector: np.array
            Вхідний вектор ознак (бінаризований).

        Returns:
        class_prediction: int
            Найбільш ймовірний клас.
        """
        best_score = None
        best_class = None

        print("\n--- Процес розпізнавання ---")

        for class_label, mean_vector in enumerate(self.class_means):
            if class_label not in self.optimal_radii or self.optimal_radii[class_label] is None:
                print(f"Клас {class_label} пропущений: немає оптимального радіуса")
                continue  # Пропускаємо, якщо немає оптимального радіуса

            radius = self.optimal_radii[class_label]
            distance = self.hamming_distance(mean_vector, test_vector)

            if radius == 0:  # Уникаємо ділення на нуль
                print(f"Клас {class_label} пропущений: радіус = 0")
                continue

            score = 1 - (distance / radius)

            print(f"Клас {class_label}: Відстань={distance}, Радіус={radius}, f={score:.3f}")

            if best_score is None or score > best_score:
                best_score = score
                best_class = class_label

        return best_class

        # --- ТЕСТОВИЙ ЗАПУСК ---
if __name__ == "__main__":
    np.random.seed(13)

    # Генеруємо навчальні дані (3 класи, по 10 векторів, 10 ознак)
    num_classes = 4
    num_vectors = 30
    num_features = 30

    train_features = np.random.randint(0, 2, (num_classes * num_vectors, num_features))
    train_labels = np.array([i for i in range(num_classes) for _ in range(num_vectors)])

    # Створюємо і тренуємо класифікатор
    ieit_classifier = IEITClassifier()
    ieit_classifier.fit(train_features, train_labels)

    # Тестове передбачення на випадковому вхідному векторі
    test_vector = np.random.randint(0, 2, num_features)
    predicted_class = ieit_classifier.predict(test_vector)

