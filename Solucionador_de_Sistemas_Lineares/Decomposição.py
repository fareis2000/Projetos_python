import matplotlib.pyplot as plt

def decompor(A, b):
    """
    Resolve o sistema linear A*x = b utilizando a decomposição LU.
    Além disso, gera um gráfico da convergência (erro relativo).

    Parâmetros:
      A : lista de listas (matriz dos coeficientes), deve ser quadrada.
      b : lista (vetor dos termos independentes).

    Retorna:
      x : vetor solução,
      L : matriz triangular inferior,
      U : matriz triangular superior.
    """
    n = len(A)
    A_mod = [row[:] for row in A]

    # BLOCO 2: Triangularização da matriz (armazenando os multiplicadores em A_mod)
    errors = []  # Lista para armazenar os erros
    for i in range(n - 1):
        for k in range(i + 1, n):
            m = A_mod[k][i] / A_mod[i][i]
            for j in range(i, n):
                A_mod[k][j] = A_mod[k][j] - m * A_mod[i][j]
            A_mod[k][i] = m  # Armazena o multiplicador que fará parte de L

        # Calculando o erro relativo após cada etapa de escalonamento
        erro_atual = max(abs(A_mod[k][i] - A_mod[k][i]) for k in range(n))
        errors.append(erro_atual)

    # BLOCO 3: Montagem das matrizes L e U
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            L[j][i] = A_mod[j][i]  # Para j >= i, os multiplicadores (ou o próprio pivô na diagonal)
            U[i][j] = A_mod[i][j]  # Elementos da linha i formam a linha de U
        L[i][i] = 1.0  # A diagonal de L é 1

    # Verifica se há zero na diagonal de U (significa que a matriz não é invertível)
    for i in range(n):
        if U[i][i] == 0:
            raise ValueError(f"A matriz é singular. O elemento U[{i}][{i}] é zero!")

    # BLOCO 4: Resolução dos sistemas triangulares

    # 4.1. Resolução do sistema Lower (L * y = b) por substituição progressiva
    y = [0.0 for _ in range(n)]
    y[0] = b[0]  # Como L[0][0] = 1
    for i in range(1, n):
        soma = 0.0
        for j in range(0, i):
            soma += L[i][j] * y[j]
        y[i] = b[i] - soma  # Como L[i][i] = 1, a divisão não é necessária

    # 4.2. Resolução do sistema Upper (U * x = y) por substituição regressiva
    x = [0.0 for _ in range(n)]
    x[n - 1] = y[n - 1] / U[n - 1][n - 1]  # Pode gerar erro se U[n-1][n-1] for zero
    for i in range(n - 2, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
        x[i] = (y[i] - soma) / U[i][i]

    # Gerando o gráfico de convergência (erro relativo)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='b')
    plt.yscale('log')  # Usando escala logarítmica no eixo y
    plt.xlabel('Passos de Escalonamento')
    plt.ylabel('Erro Relativo')
    plt.title('Convergência da Decomposição LU')
    plt.grid(True)
    plt.show()

    return x, L, U