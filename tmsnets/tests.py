import pytest
import tracemalloc
import time
import ipytest

def get_points_check_test(b, t, m, s):
    print(f"Валидная сеть (t={t}, m={m}, s={s}, b={b})")
    getpoints = get_points_opt(b, t, m, s)

    # Старт замеров
    tracemalloc.start()
    start_time = time.time()

    # Вызов функции
    result = check_tms_network(getpoints, t, m, s, b)

    # Фиксация результатов
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nРезультат: {result}")
    print(f"Время: {end_time - start_time:.2f} сек")
    print(f"Пиковая память: {peak / 1024**2:.2f} MiB\n")

    return result

def generate_random_points(b: int, m: int, s: int, seed: int = None) -> np.ndarray:

    if seed is not None:
        np.random.seed(seed)
    
    num_points = b ** m
    # Генерируем случайные точки с равномерным распределением
    points = np.random.uniform(0, 1, size=(num_points, s))
    
    # Добавляем точки в углы и центре для лучшего покрытия
    corners = np.array(list(product([0, 1], repeat=s)))
    center = np.full(s, 0.5)
    
    # Заменяем первые точки на углы
    points[:len(corners)] = corners
    
    # Добавляем центральную точку, если есть место
    if len(corners) < num_points:
        points[len(corners)] = center
    
    return points

def get_points_check_test_n(b, t, m, s):
    print(f"Случайные точки, проверка на сеть (t={t}, m={m}, s={s}, b={b})")
    getpoints = generate_random_points(b, m, s, 31)

    # Старт замеров
    tracemalloc.start()
    start_time = time.time()

    # Вызов функции
    result = check_tms_network(getpoints, t, m, s, b)

    # Фиксация результатов
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Результат: {result}")
    print(f"Время: {end_time - start_time:.2f} сек")
    print(f"Пиковая память: {peak / 1024**2:.2f} MiB\n")

    return result


test_cases_tms_checker = [
    (2, 1, 2, 2),
    (3, 0, 3, 2),
    (7, 1, 2, 3),
    (7, 1, 3, 2),
    (3, 0, 5, 2),
    (27, 1, 2, 2),
    (9, 1, 3, 3),
    (9, 1, 3, 2),
    (13, 1, 3, 2),
    (7, 2, 4, 2),
    (7, 1, 3, 5),
    (13, 0, 3, 2),
    (49, 0, 2, 3),
    (64, 1, 2, 3),
    (7, 0, 4, 2),
    (13, 1, 3, 3),
    (81, 1, 2, 3),
    (64, 0, 2, 3),
    (81, 0, 2, 3),
    (25, 1, 3, 2),
    (27, 1, 3, 2),
    (11, 0, 4, 2),
    (11, 1, 4, 3),
    (37, 0, 3, 2),
    (32, 1, 3, 3),
    (29, 0, 3, 3),
    (8, 1, 5, 3),
    (29, 0, 3, 4),
    (47, 1, 3, 3),
    (17, 1, 4, 3),
    (29, 0, 3, 5),
    (31, 0, 3, 5),
    (37, 1, 3, 5),
    (61, 1, 3, 3),
    (19, 0, 4, 3),
    (25, 1, 4, 2),
    (8, 0, 5, 4),
    (37, 0, 3, 5),
    (19, 1, 4, 3),
    (7, 1, 6, 3),
]

@pytest.mark.parametrize("b,t,m,s", [case[:4] for case in test_cases_tms_checker])
def test_tms_valid(b,t,m,s):
    assert get_points_check_test(b, t, m, s) == True, f"Ошибка для ({t}, {m}, {s}, {b})"

@pytest.mark.parametrize("b,t,m,s", [case[:4] for case in test_cases_tms_checker])
def test_tms_nonvalid(b, t, m, s):
    assert get_points_check_test_n(b, t, m, s) == False, f"Ошибка для ({t}, {m}, {s}, {b})"


def compute_t_test(b, t, m, s):
  G = generate_generator_matrices(b, t, m, s)
  res = compute_t(G, b)
  print(f"Генерация: сеть (t={t}, m={m}, s={s}, b={b})\nИтоговое t={res}")
  if t != res:
    check = check_tms_network(get_points_opt_custom(G, b, m, s), res, m, s, b)
    print(f"Проверка на сеть (t={res}, m={m}, s={s}, b={b}): {check}\n")
    return check
  print("\n")
  return True

test_cases_compute_t = [
    (3, 1, 3, 2),
    (3, 4, 5, 2),
    (3, 3, 3, 2),
    (3, 5, 5, 3),
    (3, 2, 3, 2),
    (3, 3, 5, 2),
    (3, 4, 4, 3),
    (3, 4, 5, 2),
    (5, 1, 3, 2),
    (5, 2, 5, 2),
    (5, 3, 3, 2),
    (5, 5, 5, 3),
    (5, 2, 3, 2),
    (5, 3, 5, 2),
    (5, 3, 4, 3),
    (5, 4, 5, 2),
    (7, 1, 2, 3),
    (7, 1, 3, 2),
    (7, 3, 4, 2),
    (7, 2, 3, 5),
    (7, 4, 4, 2),
    (7, 1, 2, 3),
    (7, 3, 3, 2),
    (7, 4, 4, 2),
    (7, 3, 4, 2),
    (7, 3, 3, 5),
    (8, 2, 2, 3),
    (8, 2, 3, 2),
    (8, 3, 4, 2),
    (8, 3, 3, 5),
    (8, 4, 4, 2),
    (8, 2, 2, 3),
    (8, 3, 3, 2),
    (8, 4, 4, 2),
    (9, 3, 3, 3),
    (9, 2, 3, 2),
    (9, 4, 5, 3),
    (9, 3, 3, 2),
    (9, 1, 3, 3),
    (9, 1, 4, 2),
    (9, 3, 3, 3),
    (9, 2, 3, 2),
    (13, 3, 3, 2),
    (13, 1, 3, 2),
    (13, 2, 3, 2),
    (13, 2, 3, 4),
    (13, 3, 3, 2),
    (13, 2, 3, 4),
    (13, 3, 3, 2),
    (13, 2, 3, 3),
    (25, 1, 3, 2),
    (25, 2, 3, 3),
    (25, 2, 3, 2),
    (25, 3, 3, 3),
    (27, 1, 2, 2),
    (27, 2, 2, 3),
    (27, 3, 3, 3),
    (27, 2, 2, 2),
    (49, 1, 1, 2),
    (49, 1, 2, 2),
    (49, 2, 2, 3),
    (49, 1, 3, 3),
    (64, 1, 2, 2),
    (64, 2, 2, 3),
    (64, 2, 2, 2),
    (81, 0, 2, 3),
    (81, 2, 2, 2),
    (81, 1, 2, 4),
    (81, 2, 2, 3)
]

@pytest.mark.parametrize("b,t,m,s", [case[:4] for case in test_cases_compute_t])
def test_compute_t(b, t, m, s):
    assert compute_t_test(b, t, m, s) == True, f"Ошибка для ({t}, {m}, {s}, {b})"
