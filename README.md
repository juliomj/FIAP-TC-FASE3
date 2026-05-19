# Tech Challenge Fase 3 — Machine Learning Engineering para atrasos de voos nos EUA

## 1. Contexto do problema

Este projeto foi estruturado para o Tech Challenge Fase 3 de Machine Learning Engineering. O tema é a análise de voos comerciais dos Estados Unidos, com foco em atrasos de chegada, cancelamentos, rotas críticas e padrões operacionais por companhia aérea, aeroporto e período do tempo.

A base de origem indicada no desafio contém três tabelas principais: `airlines.csv`, `airports.csv` e `flights.csv`. Como os arquivos podem ser grandes, eles não são versionados neste repositório. Para executar o projeto, os CSVs devem ser baixados e salvos localmente na pasta `data/`.

## 2. Objetivo

O objetivo é construir uma solução completa e reprodutível que responda perguntas como:

- quais companhias, aeroportos e rotas concentram mais atrasos;
- como os atrasos variam por mês, dia da semana e horário programado;
- quais características disponíveis antes do voo ajudam a prever atraso de chegada;
- se é possível agrupar aeroportos com perfis operacionais semelhantes;
- quais limitações existem na base e nos modelos.

A modelagem supervisionada foi definida como um problema de **classificação binária**: prever se um voo chegará atrasado de forma operacionalmente relevante.

## 3. Descrição das bases

### `airlines.csv`

Tabela de referência das companhias aéreas. O campo `IATA_CODE` identifica a companhia e é usado para enriquecer `flights.AIRLINE`.

### `airports.csv`

Tabela de referência dos aeroportos. O campo `IATA_CODE` é usado para enriquecer tanto o aeroporto de origem (`ORIGIN_AIRPORT`) quanto o aeroporto de destino (`DESTINATION_AIRPORT`).

### `flights.csv`

Tabela principal com registros de voos. As colunas mais importantes para este projeto são:

- `MONTH`, `DAY`, `DAY_OF_WEEK`: variáveis de calendário;
- `AIRLINE`: código da companhia aérea;
- `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`: origem e destino;
- `SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`, `SCHEDULED_TIME`: horários e duração programados;
- `DISTANCE`: distância do voo;
- `CANCELLED`, `DIVERTED`: indicadores de cancelamento e desvio;
- `ARRIVAL_DELAY`: atraso observado de chegada, usado apenas para criar a variável alvo e para EDA;
- `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`: causas de atraso disponíveis após a operação, usadas somente na análise exploratória.

O arquivo `dicionario_dados_flights (1)` pode ser colocado em `data/` para consulta durante a análise. O pipeline localiza arquivos com prefixo `dicionario_dados_flights`.

## 4. Decisão de modelagem e prevenção de data leakage

A variável alvo é:

```text
IS_DELAYED = 1 se ARRIVAL_DELAY > 15 minutos
IS_DELAYED = 0 caso contrário
```

O limite de 15 minutos foi usado por ser uma definição operacional comum para atraso relevante, evitando tratar pequenas variações como problema crítico.

Para simular uma previsão antes do voo acontecer, o modelo **não usa** variáveis que só são conhecidas depois da partida ou chegada, como `ARRIVAL_DELAY`, `ARRIVAL_TIME`, `ELAPSED_TIME`, `AIR_TIME`, `TAXI_IN`, `WHEELS_ON` e colunas de causa de atraso. Essas variáveis permanecem disponíveis apenas para EDA e interpretação do fenômeno.

As features preditivas priorizadas são variáveis conhecidas no planejamento ou antes da decolagem: calendário, companhia, origem, destino, horários programados, tempo programado, distância, período do dia, categoria de distância e indicação de rota movimentada.

## 5. Estrutura do projeto

```text
.
├── data/
│   └── README.md
├── notebooks/
│   └── tech_challenge_fase_03_voos.ipynb
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   ├── config.py
│   ├── eda.py
│   ├── features.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── run_pipeline.py
│   ├── supervised_modeling.py
│   ├── unsupervised_modeling.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 6. Como executar

### 6.1. Preparar os dados

Baixe os arquivos do desafio e salve na pasta `data/` com exatamente estes nomes:

```text
data/airlines.csv
data/airports.csv
data/flights.csv
```

Opcionalmente, salve também o dicionário de dados em `data/`.

### 6.2. Criar ambiente virtual e instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 6.3. Rodar o pipeline por script

Para executar uma amostra inicial, adequada para computadores pessoais:

```bash
python -m src.run_pipeline --sample-size 250000
```

Para executar com a base completa:

```bash
python -m src.run_pipeline --sample-size 0
```

O pipeline salva gráficos em `outputs/figures/` e tabelas de métricas em `outputs/reports/`.

### 6.4. Rodar o notebook

```bash
jupyter notebook notebooks/tech_challenge_fase_03_voos.ipynb
```

O notebook usa os mesmos módulos da pasta `src/`, portanto a lógica fica organizada e reutilizável.

## 7. Principais etapas analíticas

### 7.1. Leitura e validação

O projeto valida:

- quantidade de linhas e colunas;
- nomes e tipos das colunas;
- valores ausentes;
- duplicidades;
- proporção de cancelamentos e desvios;
- distribuição de `ARRIVAL_DELAY`;
- merges com companhias e aeroportos sem perda indevida de linhas.

### 7.2. EDA

A EDA inclui estatísticas descritivas e gráficos para:

- distribuição de atrasos de chegada;
- atrasos médios e taxa de atraso por companhia;
- aeroportos de origem críticos combinando volume e atraso;
- rotas críticas;
- padrões por mês, dia da semana e período do dia;
- cancelamentos e motivos de cancelamento;
- causas de atraso pós-voo.

### 7.3. Tratamento dos dados

Para a modelagem de atraso de chegada, voos cancelados e desviados são removidos, pois não possuem a mesma interpretação de chegada regular ao destino. Valores ausentes em variáveis essenciais são removidos ou imputados dentro de pipelines do scikit-learn. Horários no formato HHMM são convertidos para minutos desde meia-noite, e são criadas variáveis derivadas como `ROUTE`, `DEPARTURE_PERIOD`, `DISTANCE_CATEGORY`, `IS_BUSY_ROUTE` e `IS_DELAYED`.

### 7.4. Modelagem supervisionada

São comparados três modelos:

1. baseline de classe majoritária;
2. Regressão Logística;
3. Random Forest Classifier.

As métricas avaliadas são accuracy, precision, recall, F1-score, ROC-AUC, matriz de confusão e curva ROC. A acurácia não é analisada isoladamente, porque pode mascarar desempenho ruim na classe de interesse quando há desbalanceamento entre voos atrasados e não atrasados.

### 7.5. Modelagem não supervisionada

A abordagem não supervisionada clusteriza aeroportos de origem. Cada aeroporto é descrito por variáveis agregadas, como total de voos, atraso médio, taxa de atraso, distância média, taxa de cancelamento, quantidade de destinos atendidos e taxas de atraso por período do dia. O projeto usa `StandardScaler`, `KMeans`, método do cotovelo, silhouette score e PCA em duas dimensões para visualização.

## 8. Principais resultados esperados após execução

Após a execução com os CSVs locais, o repositório produzirá:

- ranking de companhias com maior atraso médio e maior taxa de atrasos;
- aeroportos críticos combinando volume e risco de atraso;
- rotas com alta concentração de atraso;
- padrões temporais por mês, dia da semana e período do dia;
- tabela comparativa de métricas dos modelos supervisionados;
- matriz de confusão e curva ROC do melhor modelo supervisionado;
- importância de variáveis da Random Forest;
- clusters de aeroportos com perfis operacionais semelhantes.

> Observação: este ambiente de entrega não continha os CSVs originais em `data/`, por isso os resultados numéricos não foram inventados no README. Eles são gerados automaticamente ao rodar o pipeline com os dados do desafio.

## 9. Limitações

- Os modelos usam variáveis disponíveis antes do voo e, por isso, não capturam totalmente eventos operacionais de última hora, clima em tempo real, congestionamento no aeroporto ou manutenção.
- A base contém variáveis de causa de atraso, mas elas representam informação pós-evento e não devem ser usadas para previsão pré-voo.
- O split implementado é aleatório estratificado. Um split temporal seria mais rigoroso para simular uso em produção.
- Agregações históricas por companhia, aeroporto ou rota devem ser calculadas apenas no conjunto de treino para evitar vazamento. Esta versão usa principalmente features diretas e uma flag simples de rota movimentada baseada em volume.
- Resultados podem variar se o pipeline for executado em amostra em vez da base completa.

## 10. Próximos passos

- Implementar validação temporal, treinando em meses anteriores e testando em meses posteriores.
- Adicionar dados externos de clima e feriados.
- Testar modelos de boosting, como Gradient Boosting ou XGBoost, se o ambiente permitir.
- Calibrar probabilidades para apoiar decisões operacionais.
- Criar agregações históricas com cálculo fold-aware ou train-only.
- Monitorar drift temporal em produção.

## 11. Checklist de atendimento aos requisitos

- EDA com estatísticas descritivas: atendido.
- Visualizações com insights: atendido.
- Tratamento de valores ausentes: atendido.
- Modelagem supervisionada: atendido.
- Comparação de dois algoritmos: atendido.
- Métricas adequadas: atendido.
- Modelagem não supervisionada: atendido.
- Gráficos e interpretação: atendido.
- Discussão crítica: atendido.
- Próximos passos: atendido.


## 12. Link do repositório GitHub

Preencher após publicação:

```text
https://github.com/<usuario>/<repositorio>
```
