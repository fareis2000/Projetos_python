import matplotlib.pyplot as plt


def gauss_eliminacao(MAT, VET):
    """
    Resolve o sistema linear MAT * X = VET usando eliminação de Gauss com retrossubstituição.
    Além disso, gera um gráfico da convergência (erro relativo).

    Parâmetros:
      MAT : lista de listas com os coeficientes da matriz (sistema quadrado).
      VET : lista com os termos independentes.

    Retorna:
      VET_SOLUCAO : lista com a solução do sistema.
    """
    NL = len(MAT)  # número de equações
    NC = len(MAT[0])  # número de incógnitas

    # BLOCO 2: Escalonamento (eliminação)
    errors = []  # Lista para armazenar os erros entre iterações
    for k in range(0, NL - 1):
        for i in range(k + 1, NL):
            M = MAT[i][k] / MAT[k][k]
            MAT[i][k] = 0  # zera o elemento abaixo do pivô
            for j in range(k + 1, NC):
                MAT[i][j] = MAT[i][j] - M * MAT[k][j]
            VET[i] = VET[i] - M * VET[k]

        # Calculando erro relativo após cada etapa de escalonamento
        erro_atual = max(abs(VET[i] - MAT[i][i]) for i in range(NL))
        errors.append(erro_atual)

    # BLOCO 3: Retrossubstituição
    VET_SOLUCAO = [0 for _ in range(NL)]
    VET_SOLUCAO[NL - 1] = VET[NL - 1] / MAT[NL - 1][NC - 1]
    for k in range(NL - 2, -1, -1):
        SOMA = 0
        for j in range(k + 1, NC):
            SOMA += MAT[k][j] * VET_SOLUCAO[j]
        VET_SOLUCAO[k] = (VET[k] - SOMA) / MAT[k][k]

    # Gerando o gráfico de convergência (erro relativo)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='g')
    plt.yscale('log')  # Usando escala logarítmica no eixo y
    plt.xlabel('Passos de Escalonamento')
    plt.ylabel('Erro Relativo')
    plt.title('Convergência da Eliminação de Gauss')
    plt.grid(True)
    plt.show()

    return VET_SOLUCAO