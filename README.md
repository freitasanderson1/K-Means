# K-Means


A tarefa deste teste diagnóstico consiste em utilizar o algoritmo K-Means a partir de uma base de dados para problemas de agrupamento (clustering). 

O conjunto de dados utilizado foi o [Wholesale customers Data Set](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set). 

Como métrica de avaliação da qualidade do aprendizado, utilize o coeficiente de silhueta.

Resumo do que o código faz:
- Carrega e filtra os dados contínuos.
- Normaliza as variáveis para evitar distorções nos clusters.
- Testa diferentes números de clusters para determinar o valor ideal usando o coeficiente de silhueta.
- Ajusta o modelo com o melhor K e analisa a composição dos clusters.
- Avalia a distribuição das variáveis categóricas nos clusters e gera gráficos para interpretar os resultados.