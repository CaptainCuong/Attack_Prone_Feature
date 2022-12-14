# Attack-Prone Feature Experiment
For each dataset, I randomly trained on 1000 sentences in the train set and randomly tested on 100 sentences in the test set. Then I randomly selected 100 sentences in the test set to attack the model with the PWWS attacker.
# Results Overview

## Attack Success Rate
| MODEL\DATASET | ag_news | amazon_review_full | amazon_review_polarity | dbpedia | sogou_news | yahoo_answers | yelp_review_full | yelp_review_polarity |
|---------------|---------|--------------------|------------------------|---------|------------|---------------|------------------|----------------------|
| Character CNN | 0.63    | 0.86               | 0.5                    | 0.93    | X          | 0.72          | 0.71             | 0.38                 |
| Word CNN      | 0.67    | 0.8                | 0.43                   | 0.67    | X          | 0.81          | X                | 0.7                  |
| Bert-base     | 0.82    | 0.88               | 0.12                   | 0       | X          | 0.46          | 0.43             | 0.37                 |
| Roberta-base  | 0.03    | 0.01               | 0                      | 0.02    | 0          | 0.31          | 0.31             | 0.22                 |
| BiLSTM        | 0.87    | 0.62               | 0.57                   | 0.81    | 0.13       | 0.91          | X                | X                    |
| LSTM          | 0.65    | 0.75               | 0.7                    | 0.84    | 0.2        | 0.68          | X                | X                    |
| RNN           | 0.73    | 0.77               | 0.68                   | 0.79    | 0.44       | 0.89          | X                | X                    |
| BiRNN         | 0.86    | 0.83               | 0.72                   | 0.83    | 0.28       | 0.87          | X                | X                    |


# Character CNN
## Traning Configuration
|       Batch Size       |        32        |
|:----------------------:|:----------------:|
| Number of tran samples | 1000             |
| Number of test samples |        100       |
|         Epoches        |        100       |
|      Learning rate     |       5e-4       |
| Sentence Length        | 1024 (character) |
## ag_news
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 63       |
| Attack Success Rate:            | 0.63     |
| Avg. Running Time:              | 0.035337 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 196.18   |
| Avg. Fluency (ppl):             | 331.27   |
| Avg. Grammatical Errors:        | 10.73    |
| Avg. Semantic Similarity:       | 0.89976  |
| Avg. Levenshtein Edit Distance: | 9.1111   |
| Avg. Word Modif. Rate:          | 0.24188  |
## amazon_review_full
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 86       |
| Attack Success Rate:            | 0.86     |
| Avg. Running Time:              | 0.046623 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 517.5    |
| Avg. Fluency (ppl):             | 117.08   |
| Avg. Grammatical Errors:        | 19.849   |
| Avg. Semantic Similarity:       | 0.95675  |
| Avg. Levenshtein Edit Distance: | 12.163   |
| Avg. Word Modif. Rate:          | 0.16075  |
## amazon_review_polarity
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 50       |
| Attack Success Rate:            | 0.5      |
| Avg. Running Time:              | 0.038393 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 489.46   |
| Avg. Fluency (ppl):             | 116.68   |
| Avg. Grammatical Errors:        | 23.12    |
| Avg. Semantic Similarity:       | 0.95378  |
| Avg. Levenshtein Edit Distance: | 14.94    |
| Avg. Word Modif. Rate:          | 0.20819  |
## dbpedia
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 93       |
| Attack Success Rate:            | 0.93     |
| Avg. Running Time:              | 0.028483 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 250.29   |
| Avg. Fluency (ppl):             | 161.87   |
| Avg. Grammatical Errors:        | 15.398   |
| Avg. Semantic Similarity:       | 0.9616   |
| Avg. Levenshtein Edit Distance: | 17.065   |
| Avg. Word Modif. Rate:          | 0.31076  |
## sogou_news
## yahoo_answers
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 72       |
| Attack Success Rate:            | 0.72     |
| Avg. Running Time:              | 0.027664 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 272.14   |
| Avg. Fluency (ppl):             | 2370.7   |
| Avg. Grammatical Errors:        | 11.528   |
| Avg. Semantic Similarity:       | 0.89899  |
| Avg. Levenshtein Edit Distance: | 6.9306   |
| Avg. Word Modif. Rate:          | 0.18228  |
## yelp_review_full
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 71       |
| Attack Success Rate:            | 0.71     |
| Avg. Running Time:              | 0.073046 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 1062.2   |
| Avg. Fluency (ppl):             | 133.65   |
| Avg. Grammatical Errors:        | 37.887   |
| Avg. Semantic Similarity:       | 0.96549  |
| Avg. Levenshtein Edit Distance: | 20.873   |
| Avg. Word Modif. Rate:          | 0.14237  |
## yelp_review_polarity
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 38       |
| Attack Success Rate:            | 0.38     |
| Avg. Running Time:              | 0.093736 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 929.41   |
| Avg. Fluency (ppl):             | 171.14   |
| Avg. Grammatical Errors:        | 27.737   |
| Avg. Semantic Similarity:       | 0.92452  |
| Avg. Levenshtein Edit Distance: | 16.684   |
| Avg. Word Modif. Rate:          | 0.14807  |
# Word CNN
## ag_news
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 67      |
| Attack Success Rate:            | 0.67    |
| Avg. Running Time:              | 0.01234 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 187.12  |
| Avg. Fluency (ppl):             | 340.4   |
| Avg. Grammatical Errors:        | 7.3582  |
| Avg. Semantic Similarity:       | 0.94288 |
| Avg. Levenshtein Edit Distance: | 7.9701  |
| Avg. Word Modif. Rate:          | 0.244   |
## amazon_review_full
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 80       |
| Attack Success Rate:            | 0.8      |
| Avg. Running Time:              | 0.024652 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 512.63   |
| Avg. Fluency (ppl):             | 115.04   |
| Avg. Grammatical Errors:        | 19.325   |
| Avg. Semantic Similarity:       | 0.95634  |
| Avg. Levenshtein Edit Distance: | 11.363   |
| Avg. Word Modif. Rate:          | 0.14546  |
## amazon_review_polarity
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 43       |
| Attack Success Rate:            | 0.43     |
| Avg. Running Time:              | 0.025495 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 499.65   |
| Avg. Fluency (ppl):             | 144.82   |
| Avg. Grammatical Errors:        | 19.767   |
| Avg. Semantic Similarity:       | 0.93881  |
| Avg. Levenshtein Edit Distance: | 12.535   |
| Avg. Word Modif. Rate:          | 0.16122  |
## dbpedia
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 67       |
| Attack Success Rate:            | 0.67     |
| Avg. Running Time:              | 0.013617 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 241.65   |
| Avg. Fluency (ppl):             | 174.33   |
| Avg. Grammatical Errors:        | 16.478   |
| Avg. Semantic Similarity:       | 0.9458   |
| Avg. Levenshtein Edit Distance: | 18.149   |
| Avg. Word Modif. Rate:          | 0.32841  |
## sogou_news
## yahoo_answers
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 81       |
| Attack Success Rate:            | 0.81     |
| Avg. Running Time:              | 0.013024 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 217.63   |
| Avg. Fluency (ppl):             | 822.04   |
| Avg. Grammatical Errors:        | 8.2222   |
| Avg. Semantic Similarity:       | 0.86665  |
| Avg. Levenshtein Edit Distance: | 4.6173   |
| Avg. Word Modif. Rate:          | 0.17336  |
## yelp_review_full
## yelp_review_polarity
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 70       |
| Attack Success Rate:            | 0.7      |
| Avg. Running Time:              | 0.064513 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 1074.8   |
| Avg. Fluency (ppl):             | 94.452   |
| Avg. Grammatical Errors:        | 32.614   |
| Avg. Semantic Similarity:       | 0.96492  |
| Avg. Levenshtein Edit Distance: | 17.2     |
| Avg. Word Modif. Rate:          | 0.12796  |
# Bert-base
## ag_news
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 82      |
| Attack Success Rate:            | 0.82    |
| Avg. Running Time:              | 0.0612  |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 183.42  |
| Avg. Fluency (ppl):             | 492.2   |
| Avg. Grammatical Errors:        | 9.0976  |
| Avg. Semantic Similarity:       | 0.89378 |
| Avg. Levenshtein Edit Distance: | 10.293  |
| Avg. Word Modif. Rate:          | 0.30092 |
## amazon_review_full
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 88       |
| Attack Success Rate:            | 0.88     |
| Avg. Running Time:              | 0.088293 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 523.1    |
| Avg. Fluency (ppl):             | 178.6    |
| Avg. Grammatical Errors:        | 20.114   |
| Avg. Semantic Similarity:       | 0.90382  |
| Avg. Levenshtein Edit Distance: | 14.739   |
| Avg. Word Modif. Rate:          | 0.26033  |
## amazon_review_polarity
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 12      |
| Attack Success Rate:            | 0.12    |
| Avg. Running Time:              | 0.31382 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 581.08  |
| Avg. Fluency (ppl):             | 121.99  |
| Avg. Grammatical Errors:        | 23.583  |
| Avg. Semantic Similarity:       | 0.92965 |
| Avg. Levenshtein Edit Distance: | 18.917  |
| Avg. Word Modif. Rate:          | 0.20811 |
## dbpedia
|               Summary|                |
| ------ | ------ |
| Total Attacked Instances:  | 100     |
| Successful Instances:      | 0       |
| Attack Success Rate:       | 0       |
| Avg. Running Time:         | 0.23207 |
| Total Query Exceeded:      | 0       |
| Avg. Victim Model Queries: | 237.29  |
## sogou_news
## yahoo_answers
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 46      |
| Attack Success Rate:            | 0.46    |
| Avg. Running Time:              | 0.09895 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 244.12  |
| Avg. Fluency (ppl):             | 1095.9  |
| Avg. Grammatical Errors:        | 10.217  |
| Avg. Semantic Similarity:       | 0.79672 |
| Avg. Levenshtein Edit Distance: | 6.9783  |
| Avg. Word Modif. Rate:          | 0.25843 |
## yelp_review_full
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 43      |
| Attack Success Rate:            | 0.43    |
| Avg. Running Time:              | 0.39631 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 943.95  |
| Avg. Fluency (ppl):             | 274.61  |
| Avg. Grammatical Errors:        | 29.326  |
| Avg. Semantic Similarity:       | 0.8389  |
| Avg. Levenshtein Edit Distance: | 28.907  |
| Avg. Word Modif. Rate:          | 0.30585 |
## yelp_review_polarity
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 37      |
| Attack Success Rate:            | 0.37    |
| Avg. Running Time:              | 0.59489 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 1123.8  |
| Avg. Fluency (ppl):             | 265.19  |
| Avg. Grammatical Errors:        | 16.73   |
| Avg. Semantic Similarity:       | 0.88568 |
| Avg. Levenshtein Edit Distance: | 15.622  |
| Avg. Word Modif. Rate:          | 0.21581 |
# Roberta-base
## ag_news
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 3       |
| Attack Success Rate:            | 0.03    |
| Avg. Running Time:              | 0.17399 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 214.7   |
| Avg. Fluency (ppl):             | 481.11  |
| Avg. Grammatical Errors:        | 7.3333  |
| Avg. Semantic Similarity:       | 0.89451 |
| Avg. Levenshtein Edit Distance: | 8.3333  |
| Avg. Word Modif. Rate:          | 0.54254 |
## amazon_review_full
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 1       |
| Attack Success Rate:            | 0.01    |
| Avg. Running Time:              | 0.30209 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 558.37  |
| Avg. Fluency (ppl):             | 86.099  |
| Avg. Grammatical Errors:        | 27      |
| Avg. Semantic Similarity:       | 0.93297 |
| Avg. Levenshtein Edit Distance: | 27      |
| Avg. Word Modif. Rate:          | 0.75325 |
## amazon_review_polarity
|               Summary|                |
| ------ | ------ |
| Total Attacked Instances:  | 100     |
| Successful Instances:      | 0       |
| Attack Success Rate:       | 0       |
| Avg. Running Time:         | 0.33149 |
| Total Query Exceeded:      | 0       |
| Avg. Victim Model Queries: | 624.26  |
## dbpedia
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 2       |
| Attack Success Rate:            | 0.02    |
| Avg. Running Time:              | 0.20821 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 248.63  |
| Avg. Fluency (ppl):             | 190.8   |
| Avg. Grammatical Errors:        | 5.5     |
| Avg. Semantic Similarity:       | 0.92398 |
| Avg. Levenshtein Edit Distance: | 3.5     |
| Avg. Word Modif. Rate:          | 0.20238 |
## sogou_news
|               Summary|               |
| ------ | ------ |
| Total Attacked Instances:  | 100    |
| Successful Instances:      | 0      |
| Attack Success Rate:       | 0      |
| Avg. Running Time:         | 2.8389 |
| Total Query Exceeded:      | 0      |
| Avg. Victim Model Queries: | 1045.1 |
## yahoo_answers
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 31      |
| Attack Success Rate:            | 0.31    |
| Avg. Running Time:              | 0.10891 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 325.3   |
| Avg. Fluency (ppl):             | 220.26  |
| Avg. Grammatical Errors:        | 25.226  |
| Avg. Semantic Similarity:       | 0.82606 |
| Avg. Levenshtein Edit Distance: | 20.581  |
| Avg. Word Modif. Rate:          | 0.27474 |
## yelp_review_full
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 31      |
| Attack Success Rate:            | 0.31    |
| Avg. Running Time:              | 0.38577 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 1138.2  |
| Avg. Fluency (ppl):             | 126.79  |
| Avg. Grammatical Errors:        | 62.452  |
| Avg. Semantic Similarity:       | 0.90426 |
| Avg. Levenshtein Edit Distance: | 51.742  |
| Avg. Word Modif. Rate:          | 0.4104  |
## yelp_review_polarity
|                  Summary|                  |
| ------ | ------ |
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 22      |
| Attack Success Rate:            | 0.22    |
| Avg. Running Time:              | 0.4549  |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 984.76  |
| Avg. Fluency (ppl):             | 215.4   |
| Avg. Grammatical Errors:        | 32.591  |
| Avg. Semantic Similarity:       | 0.8901  |
| Avg. Levenshtein Edit Distance: | 27.773  |
| Avg. Word Modif. Rate:          | 0.52619 |
# BiLSTM
## Traning Configuration
|       Batch Size       |  32  |
|:----------------------:|:----:|
| Number of tran samples | 1000 |
| Number of test samples |  100 |
|         Epoches        |  150 |
|      Learning rate     | 5e-4 |
| Sentence Length        | 10   |
## ag_news
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 87        |
| Attack Success Rate:            | 0.87      |
| Avg. Running Time:              | 0.0072137 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 48.54     |
| Avg. Fluency (ppl):             | 16284     |
| Avg. Grammatical Errors:        | 2.2529    |
| Avg. Semantic Similarity:       | 0.83014   |
| Avg. Levenshtein Edit Distance: | 4.1724    |
| Avg. Word Modif. Rate:          | 0.60395   |
## amazon_review_full
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 62        |
| Attack Success Rate:            | 0.62      |
| Avg. Running Time:              | 0.0085613 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 37.04     |
| Avg. Fluency (ppl):             | 5983.8    |
| Avg. Grammatical Errors:        | 1.371     |
| Avg. Semantic Similarity:       | 0.71345   |
| Avg. Levenshtein Edit Distance: | 2.5968    |
| Avg. Word Modif. Rate:          | 0.68857   |
## amazon_review_polarity
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 57        |
| Attack Success Rate:            | 0.57      |
| Avg. Running Time:              | 0.0074409 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 36.99     |
| Avg. Fluency (ppl):             | 8623.7    |
| Avg. Grammatical Errors:        | 1.3684    |
| Avg. Semantic Similarity:       | 0.69062   |
| Avg. Levenshtein Edit Distance: | 2.7368    |
| Avg. Word Modif. Rate:          | 0.62544   |
## dbpedia
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 81        |
| Attack Success Rate:            | 0.81      |
| Avg. Running Time:              | 0.0076351 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 16.5      |
| Avg. Fluency (ppl):             | 7071.9    |
| Avg. Grammatical Errors:        | 2.0741    |
| Avg. Semantic Similarity:       | 0.77779   |
| Avg. Levenshtein Edit Distance: | 2.7407    |
| Avg. Word Modif. Rate:          | 0.834     |
## sogou_news
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 13       |
| Attack Success Rate:            | 0.13     |
| Avg. Running Time:              | 0.014523 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 29       |
| Avg. Fluency (ppl):             | 314.49   |
| Avg. Grammatical Errors:        | 6.8462   |
| Avg. Semantic Similarity:       | 0.95363  |
| Avg. Levenshtein Edit Distance: | 0.46154  |
| Avg. Word Modif. Rate:          | 0.054654 |
## yahoo_answers
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 91       |
| Attack Success Rate:            | 0.91     |
| Avg. Running Time:              | 0.010651 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 77.03    |
| Avg. Fluency (ppl):             | 2241.4   |
| Avg. Grammatical Errors:        | 2.7143   |
| Avg. Semantic Similarity:       | 0.84422  |
| Avg. Levenshtein Edit Distance: | 3.0879   |
| Avg. Word Modif. Rate:          | 0.2868   |
# LSTM
## ag_news
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 65        |
| Attack Success Rate:            | 0.65      |
| Avg. Running Time:              | 0.0086237 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 51.75     |
| Avg. Fluency (ppl):             | 4305.2    |
| Avg. Grammatical Errors:        | 2.6615    |
| Avg. Semantic Similarity:       | 0.8055    |
| Avg. Levenshtein Edit Distance: | 5.1692    |
| Avg. Word Modif. Rate:          | 0.66454   |
## amazon_review_full
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 75        |
| Attack Success Rate:            | 0.75      |
| Avg. Running Time:              | 0.0085722 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 37.04     |
| Avg. Fluency (ppl):             | 26716     |
| Avg. Grammatical Errors:        | 1.72      |
| Avg. Semantic Similarity:       | 0.67169   |
| Avg. Levenshtein Edit Distance: | 2.6533    |
| Avg. Word Modif. Rate:          | 0.57327   |
## amazon_review_polarity
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 70       |
| Attack Success Rate:            | 0.7      |
| Avg. Running Time:              | 0.010094 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 36.79    |
| Avg. Fluency (ppl):             | 7755.1   |
| Avg. Grammatical Errors:        | 1.4857   |
| Avg. Semantic Similarity:       | 0.68932  |
| Avg. Levenshtein Edit Distance: | 2.5429   |
| Avg. Word Modif. Rate:          | 0.66905  |
## dbpedia
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 84       |
| Attack Success Rate:            | 0.84     |
| Avg. Running Time:              | 0.006229 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 16.59    |
| Avg. Fluency (ppl):             | 20182    |
| Avg. Grammatical Errors:        | 1.9048   |
| Avg. Semantic Similarity:       | 0.76734  |
| Avg. Levenshtein Edit Distance: | 2.7381   |
| Avg. Word Modif. Rate:          | 0.84702  |
## sogou_news
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 20       |
| Attack Success Rate:            | 0.2      |
| Avg. Running Time:              | 0.012125 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 28.66    |
| Avg. Fluency (ppl):             | 164.88   |
| Avg. Grammatical Errors:        | 9        |
| Avg. Semantic Similarity:       | 0.9367   |
| Avg. Levenshtein Edit Distance: | 0.4      |
| Avg. Word Modif. Rate:          | 0.044097 |
## yahoo_answers
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 68        |
| Attack Success Rate:            | 0.68      |
| Avg. Running Time:              | 0.0079263 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 73.13     |
| Avg. Fluency (ppl):             | 2978.7    |
| Avg. Grammatical Errors:        | 3.0588    |
| Avg. Semantic Similarity:       | 0.82857   |
| Avg. Levenshtein Edit Distance: | 3.2794    |
| Avg. Word Modif. Rate:          | 0.30678   |
# RNN
## ag_news
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 73        |
| Attack Success Rate:            | 0.73      |
| Avg. Running Time:              | 0.0061613 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 48.94     |
| Avg. Fluency (ppl):             | 12848     |
| Avg. Grammatical Errors:        | 2.3836    |
| Avg. Semantic Similarity:       | 0.81213   |
| Avg. Levenshtein Edit Distance: | 4.8767    |
| Avg. Word Modif. Rate:          | 0.65107   |
## amazon_review_full
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 77        |
| Attack Success Rate:            | 0.77      |
| Avg. Running Time:              | 0.0048711 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 40.3      |
| Avg. Fluency (ppl):             | 13399     |
| Avg. Grammatical Errors:        | 1.6104    |
| Avg. Semantic Similarity:       | 0.7122    |
| Avg. Levenshtein Edit Distance: | 2.5974    |
| Avg. Word Modif. Rate:          | 0.5635    |
## amazon_review_polarity
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 68        |
| Attack Success Rate:            | 0.68      |
| Avg. Running Time:              | 0.0064955 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 35.06     |
| Avg. Fluency (ppl):             | nan       |
| Avg. Grammatical Errors:        | 1.4853    |
| Avg. Semantic Similarity:       | 0.71555   |
| Avg. Levenshtein Edit Distance: | 2.6618    |
| Avg. Word Modif. Rate:          | 0.6161    |
## dbpedia
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 79       |
| Attack Success Rate:            | 0.79     |
| Avg. Running Time:              | 0.005091 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 12.76    |
| Avg. Fluency (ppl):             | 13204    |
| Avg. Grammatical Errors:        | 2.2658   |
| Avg. Semantic Similarity:       | 0.83975  |
| Avg. Levenshtein Edit Distance: | 2.5063   |
| Avg. Word Modif. Rate:          | 0.82188  |
## sogou_news
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 44        |
| Attack Success Rate:            | 0.44      |
| Avg. Running Time:              | 0.0059475 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 23.27     |
| Avg. Fluency (ppl):             | 156.9     |
| Avg. Grammatical Errors:        | 9.5       |
| Avg. Semantic Similarity:       | 0.98492   |
| Avg. Levenshtein Edit Distance: | 0.22727   |
| Avg. Word Modif. Rate:          | 0.026962  |
## yahoo_answers
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 89        |
| Attack Success Rate:            | 0.89      |
| Avg. Running Time:              | 0.0086533 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 75.02     |
| Avg. Fluency (ppl):             | 2533.3    |
| Avg. Grammatical Errors:        | 2.7191    |
| Avg. Semantic Similarity:       | 0.8352    |
| Avg. Levenshtein Edit Distance: | 2.7079    |
| Avg. Word Modif. Rate:          | 0.29397   |
# BiRNN
## ag_news
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 86        |
| Attack Success Rate:            | 0.86      |
| Avg. Running Time:              | 0.0074141 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 46.99     |
| Avg. Fluency (ppl):             | 7828.8    |
| Avg. Grammatical Errors:        | 2.3837    |
| Avg. Semantic Similarity:       | 0.8273    |
| Avg. Levenshtein Edit Distance: | 4.5233    |
| Avg. Word Modif. Rate:          | 0.63394   |
## amazon_review_full
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 83       |
| Attack Success Rate:            | 0.83     |
| Avg. Running Time:              | 0.007181 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 37.37    |
| Avg. Fluency (ppl):             | 3425.7   |
| Avg. Grammatical Errors:        | 1.7349   |
| Avg. Semantic Similarity:       | 0.70067  |
| Avg. Levenshtein Edit Distance: | 2.4337   |
| Avg. Word Modif. Rate:          | 0.50821  |
## amazon_review_polarity
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 72        |
| Attack Success Rate:            | 0.72      |
| Avg. Running Time:              | 0.0078038 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 37.64     |
| Avg. Fluency (ppl):             | 24251     |
| Avg. Grammatical Errors:        | 1.7361    |
| Avg. Semantic Similarity:       | 0.73189   |
| Avg. Levenshtein Edit Distance: | 2.6806    |
| Avg. Word Modif. Rate:          | 0.61892   |
## dbpedia
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 83       |
| Attack Success Rate:            | 0.83     |
| Avg. Running Time:              | 0.005928 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 14.86    |
| Avg. Fluency (ppl):             | 10966    |
| Avg. Grammatical Errors:        | 2.0843   |
| Avg. Semantic Similarity:       | 0.78597  |
| Avg. Levenshtein Edit Distance: | 2.4819   |
| Avg. Word Modif. Rate:          | 0.80459  |
## sogou_news
|                  Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100      |
| Successful Instances:           | 28       |
| Attack Success Rate:            | 0.28     |
| Avg. Running Time:              | 0.010882 |
| Total Query Exceeded:           | 0        |
| Avg. Victim Model Queries:      | 27.47    |
| Avg. Fluency (ppl):             | 164.79   |
| Avg. Grammatical Errors:        | 8.1786   |
| Avg. Semantic Similarity:       | 0.94816  |
| Avg. Levenshtein Edit Distance: | 0.53571  |
| Avg. Word Modif. Rate:          | 0.041106 |
## yahoo_answers
|                   Summary|                   |
| ------ | ------ |
| Total Attacked Instances:       | 100       |
| Successful Instances:           | 87        |
| Attack Success Rate:            | 0.87      |
| Avg. Running Time:              | 0.0071098 |
| Total Query Exceeded:           | 0         |
| Avg. Victim Model Queries:      | 76.02     |
| Avg. Fluency (ppl):             | 2091.3    |
| Avg. Grammatical Errors:        | 2.7471    |
| Avg. Semantic Similarity:       | 0.82827   |
| Avg. Levenshtein Edit Distance: | 2.6897    |
| Avg. Word Modif. Rate:          | 0.3032    |

<!-- # BUG
IndexError: index out of range in self -->