import matplotlib.pyplot as plt


def gauss_seidel(A, b, x0, tol=0.005, N_MAX=200):
    """
    Resolve o sistema linear A * x = b utilizando o método iterativo de Gauss–Seidel e gera o gráfico de convergência.

    Parâmetros:
      A    : matriz dos coeficientes (lista de listas, quadrada);
      b    : vetor dos termos independentes (lista);
      x0   : vetor inicial (lista);
      tol  : tolerância para o critério de parada (default 0.005);
      N_MAX: número máximo de iterações (default 200).

    Retorna:
      x    : vetor solução aproximada.
    """
    n = len(A)
    k = 0
    errors = []  # Lista para armazenar os erros ao longo das iterações

    while k < N_MAX:
        xk = [0.0] * n
        for i in range(n):
            soma = 0.0
            for j in range(0, i):
                soma += A[i][j] * xk[j]
            for j in range(i + 1, n):
                soma += A[i][j] * x0[j]
            xk[i] = (b[i] - soma) / A[i][i]

        # Calculando o erro da iteração atual
        diff = max(abs(xk[i] - x0[i]) for i in range(n))
        errors.append(diff)

        # Se o erro for menor que a tolerância, interrompe
        if diff < tol:
            break

        x0 = xk[:]
        k += 1

    # Gerando o gráfico de convergência
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='b')
    plt.yscale('log')  # Usando escala logarítmica no eixo y
    plt.xlabel('Iteração')
    plt.ylabel('Erro Máximo')
    plt.title('Convergência do Método de Gauss-Seidel')
    plt.grid(True)
    plt.show()

    return xk

