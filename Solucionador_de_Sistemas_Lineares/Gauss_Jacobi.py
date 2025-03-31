import matplotlib.pyplot as plt


def gauss_jacobi(A, b, tol=0.005, N_MAX=20):
    """
    Resolve o sistema linear A * x = b utilizando o método iterativo de Jacobi e gera o gráfico de convergência.

    Parâmetros:
      A    : matriz dos coeficientes (lista de listas, quadrada);
      b    : vetor dos termos independentes (lista);
      tol  : tolerância para o critério de parada (default 0.005);
      N_MAX: número máximo de iterações (default 20).

    Retorna:
      x    : vetor solução aproximada.
    """
    n = len(A)  # número de equações (e incógnitas)
    x0 = [0.0 for _ in range(n)]  # Inicializa x0 como vetor zero
    k = 0
    errors = []  # Lista para armazenar os erros ao longo das iterações

    while k < N_MAX:
        xk = [0.0 for _ in range(n)]
        # Para cada equação i
        for i in range(n):
            soma = 0.0
            # Soma os termos j != i
            for j in range(n):
                if i != j:
                    soma -= (A[i][j] / A[i][i]) * x0[j]
            # Calcula o novo valor de x para a equação i
            xk[i] = soma + (b[i] / A[i][i])

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
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='r')
    plt.yscale('log')  # Usando escala logarítmica no eixo y
    plt.xlabel('Iteração')
    plt.ylabel('Erro Máximo')
    plt.title('Convergência do Método de Gauss-Jacobi')
    plt.grid(True)
    plt.show()

    return xk