def read_file(file_path: str) -> tuple:
    with open(file_path, 'r', encoding='UTF8') as file:
        task_type = file.readline().strip()
        n = int(file.readline().strip())
        if n <= 0:
            raise ValueError("invalid data: n must be positive")
        c = file.readline().split()
        if len(c) < n:
            raise ValueError("invalid data: недостаточно коэффициентов c")
        if task_type not in ('max', 'min'):
            raise ValueError("invalid type: task_type должен быть 'max' или 'min'")
        m = int(file.readline().strip())
        if m <= 0:
            raise ValueError("invalid data: m must be positive")
        data = file.readlines()
        if len(data) < m:
            raise ValueError("invalid data: недостаточно строк ограничений")
    return n, c, m, data, task_type


def to_canonical(n, c, m, data, task_type) -> tuple:
    if task_type == 'max':
        c = [-1.0 * float(k) for k in c]
    else:
        c = [float(k) for k in c]

    slack_count = sum(1 for row in data if row.strip().split()[n] in ('<=', '>='))
    new_n = n + slack_count
    parsed = [[0.0] * (new_n + 1) for _ in range(m)]

    k = 0
    for i in range(m):
        list_row = data[i].strip().split()
        parsed[i][:n] = map(float, list_row[:n])
        sign = list_row[n]

        if sign == '>=':
            parsed[i][n + k] = -1.0
            k += 1
        elif sign == '<=':
            parsed[i][n + k] = 1.0
            k += 1

        parsed[i][-1] = float(list_row[-1])
        if parsed[i][-1] < 0:
            parsed[i] = [-x for x in parsed[i]]

    return new_n, c, m, parsed


def extend_objective_coefficients(c, n, m) -> list:
    return list(c) + [0.0] * (n + m - len(c))


def initial_basis_and_free(n, m) -> tuple:
    basis = [n + i for i in range(m)]
    free = [j for j in range(n)]
    return basis, free


def build_auxiliary_objective(n, m, a) -> list:
    row = [0.0] * (n + 1)
    row[:-1] = [-sum(a[i][j] for i in range(m)) for j in range(n)]
    row[-1] = -sum(a[i][-1] for i in range(m))
    return row


def build_main_objective(n, m, c, a, basis, free) -> list:
    row = [0.0] * (n + 1)
    for j in range(n):
        value = sum(c[basis[i]] * a[i][j] for i in range(m))
        row[j] = -(value - c[free[j]])
    row[-1] = -sum(c[basis[i]] * a[i][-1] for i in range(m))
    return row


def choose_pivot(n, m, table) -> tuple:
    reduced_costs = table[-1][:-1]
    rhs = [table[i][-1] for i in range(m)]

    min_cost = min(reduced_costs)
    if min_cost >= -1e-9:
        return -1, -1

    pivot_col = reduced_costs.index(min_cost)
    pivot_row = -1
    for i in range(m):
        if table[i][pivot_col] > 1e-9 and rhs[i] >= 0:
            if pivot_row == -1:
                pivot_row = i
            elif rhs[i] / table[i][pivot_col] < rhs[pivot_row] / table[pivot_row][pivot_col]:
                pivot_row = i

    if pivot_row == -1:
        return -1, -1
    return pivot_row, pivot_col


def pivot_table(n, m, table, pivot_row, pivot_col, is_auxiliary) -> list:
    pivot_value = table[pivot_row][pivot_col]
    new_table = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == pivot_row and j == pivot_col:
                new_table[i][j] = 0.0 if is_auxiliary else 1.0 / pivot_value
            elif j == pivot_col:
                new_table[i][j] = 0.0 if is_auxiliary else -table[i][j] / pivot_value
            elif i == pivot_row:
                new_table[i][j] = table[i][j] / pivot_value
            else:
                new_table[i][j] = table[i][j] - table[pivot_row][j] * table[i][pivot_col] / pivot_value

    return new_table


def run_phase(n, m, table, basis, free, is_auxiliary) -> tuple:
    pivot_row, pivot_col = choose_pivot(n, m, table)
    if pivot_row == -1:
        return table, basis, free

    table = pivot_table(n, m, table, pivot_row, pivot_col, is_auxiliary)
    basis[pivot_row], free[pivot_col] = free[pivot_col], basis[pivot_row]
    return run_phase(n, m, table, basis, free, is_auxiliary)


def solve_auxiliary_problem(n, m, a, basis, free) -> tuple:
    objective_row = build_auxiliary_objective(n, m, a)
    table = a + [objective_row]
    table, basis, free = run_phase(n, m, table, basis, free, is_auxiliary=True)

    EPS = 1e-6
    if abs(table[-1][-1]) > EPS:
        raise ValueError("допустимое решение не найдено: задача несовместна")

    return table[:-1], basis, free


def solve_main_problem(n, m, c, a, basis, free) -> tuple:
    objective_row = build_main_objective(n, m, c, a, basis, free)
    table = a + [objective_row]
    return run_phase(n, m, table, basis, free, is_auxiliary=False)


def extract_solution(n_orig, table, basis, task_type) -> tuple:
    x = [0.0] * n_orig
    for i, var in enumerate(basis):
        if var < n_orig:
            x[var] = table[i][-1]

    f_val = table[-1][-1]
    if task_type == 'min':
        f_val = -f_val

    EPS_ROUND = 9
    x = [round(v, EPS_ROUND) for v in x]
    f_val = round(f_val, EPS_ROUND)
    return x, f_val


def print_solution(x, f_val) -> None:
    print("\n" + "=" * 30)
    print("РЕЗУЛЬТАТ:")
    print(f"Вектор x: {x}")
    print(f"Оптимальное значение F: {f_val}")
    print("=" * 30)


def main() -> None:
    n, c, m, data, task_type = read_file('input.txt')
    n_canon, c_raw, m, a = to_canonical(n, c, m, data, task_type)
    c_canon = extend_objective_coefficients(c_raw, n_canon, m)
    basis, free = initial_basis_and_free(n_canon, m)

    a, basis, free = solve_auxiliary_problem(n_canon, m, a, basis, free)
    final_table, basis, free = solve_main_problem(n_canon, m, c_canon, a, basis, free)

    x, f_val = extract_solution(n, final_table, basis, task_type)
    print_solution(x, f_val)


if __name__ == "__main__":
    main()