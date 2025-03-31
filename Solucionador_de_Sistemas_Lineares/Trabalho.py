import re
from Elimin_Gaus import gauss_eliminacao
from Decomposição import decompor
from Gauss_Jacobi import gauss_jacobi
from Gauss_Seidel import gauss_seidel


def Ler_Coeficientes(n):
    """
    Lê uma linha de entrada e extrai exatamente n números, permitindo espaçamentos irregulares.
    """
    while True:
        linha = input().strip()
        tokens = re.findall(r'-?\d+(?:\.\d+)?', linha)
        if len(tokens) != n:
            print(
                f"Erro: foram encontrados {len(tokens)} números, mas o esperado é {n}. Insira novamente a linha.")
        else:
            try:
                return list(map(float, tokens))
            except ValueError:
                print("Erro: entrada inválida. Tente novamente.")


def Ler_Matriz(n, Matriz=None):
    """
    Lê uma matriz de dimensões n x Matriz digitada pelo usuário.
    Se Matriz não for informado, assume uma matriz quadrada (n x n).
    """
    if Matriz is None:
        Matriz = n
    matrix = []
    for i in range(n):
        print(f"Equação {i+1}: ", end="")
        row = Ler_Coeficientes(Matriz)
        matrix.append(row)
    return matrix


def Ler_vetor(n):
    """
    Lê um vetor de n elementos (termos independentes ou chute inicial) digitados pelo usuário.
    """
    vetor = []
    for i in range(n):
        while True:
            termo_str = input(f"Valor {i+1}: ").strip()
            tokens = re.findall(r'-?\d+(?:\.\d+)?', termo_str)
            if len(tokens) != 1:
                print("Erro: insira exatamente um número.")
            else:
                try:
                    vetor.append(float(tokens[0]))
                    break
                except ValueError:
                    print("Erro: entrada inválida. Tente novamente.")
    return vetor


def gauss_mode():
    print("\nResolução de Sistema Linear por Eliminação de Gauss")
    Numero_Linhas = int(input("Digite o número de equações (linhas): "))
    Numero_Colunas = int(input("Digite o número de incógnitas (colunas): "))
    if Numero_Linhas != Numero_Colunas:
        print("Erro: o método de Gauss requer um sistema quadrado (Numero_Linhas == Numero_Colunas).")
        return
    print("\nDigite os coeficientes da matriz:")
    A = Ler_Matriz(Numero_Linhas, Numero_Colunas)
    print("\nDigite os termos independentes:")
    b = Ler_vetor(Numero_Linhas)
    solution = gauss_eliminacao(A, b)
    print("\nSolução do sistema:")
    for i, valor in enumerate(solution):
        print(f"x{i+1} = {valor}")


def lu_mode():
    print("\nResolução de Sistema Linear por Decomposição LU")
    n = int(input("Digite a dimensão da matriz (n): "))
    print("\nDigite a matriz dos coeficientes:")
    A = Ler_Matriz(n)
    print("\nDigite os termos independentes:")
    b = Ler_vetor(n)
    solution, L, U = decompor(A, b)
    print("\nSolução do sistema:")
    for i, valor in enumerate(solution):
        print(f"x{i+1} = {valor}")
    print("\nMatriz L:")
    for row in L:
        print(row)
    print("\nMatriz U:")
    for row in U:
        print(row)


def jacobi_mode():
    print("\nResolução de Sistema Linear pelo Método Iterativo de Jacobi")
    n = int(input("Digite o número de equações (n): "))
    print("\nDigite a matriz dos coeficientes:")
    A = Ler_Matriz(n)
    print("\nDigite os termos independentes:")
    b = Ler_vetor(n)
    tol_input = input(
        "Digite a tolerância (pressione Enter para usar 0.005): ")
    tol = 0.005 if tol_input == "" else float(tol_input)
    N_MAX_input = input(
        "Digite o número máximo de iterações (pressione Enter para usar 20): ")
    N_MAX = 20 if N_MAX_input == "" else int(N_MAX_input)
    solution = gauss_jacobi(A, b, tol, N_MAX)
    print("\nSolução aproximada do sistema:")
    for i, valor in enumerate(solution):
        print(f"x{i+1} = {valor}")


def gauss_seidel_mode():
    print("\nResolução de Sistema Linear pelo Método Iterativo de Gauss–Seidel")
    n = int(input("Digite o número de equações (n): "))
    print("\nDigite a matriz dos coeficientes:")
    A = Ler_Matriz(n)
    print("\nDigite os termos independentes:")
    b = Ler_vetor(n)
    print("\nDigite o vetor inicial (chute inicial):")
    x0 = Ler_vetor(n)
    tol_input = input(
        "Digite a tolerância (pressione Enter para usar 0.005): ")
    tol = 0.005 if tol_input == "" else float(tol_input)
    N_MAX_input = input(
        "Digite o número máximo de iterações (pressione Enter para usar 200): ")
    N_MAX = 200 if N_MAX_input == "" else int(N_MAX_input)
    solution = gauss_seidel(A, b, x0, tol, N_MAX)
    print("\nSolução aproximada do sistema:")
    for i, valor in enumerate(solution):
        print(f"x{i+1} = {valor}")


def main():
    while True:
        print("\nMenu de Métodos de Resolução de Sistemas Lineares:")
        print("1 - Eliminação de Gauss")
        print("2 - Decomposição LU")
        print("3 - Método Iterativo de Jacobi")
        print("4 - Método Iterativo de Gauss–Seidel")
        print("5 - Sair")
        opcao = input("Escolha a opção desejada: ").strip()
        if opcao == "1":
            gauss_mode()
        elif opcao == "2":
            lu_mode()
        elif opcao == "3":
            jacobi_mode()
        elif opcao == "4":
            gauss_seidel_mode()
        elif opcao == "5":
            print("Encerrando o programa.")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()