<!-- # Roberta-base
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity -->
For each dataset, I randomly trained on 1000 sentences in the train set and randomly tested on 100 sentences in the test set. Then I randomly selected 100 sentences in the test set to attack the model with the PWWS attacker.
# Character CNN
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
## imdb
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
<!-- # Word CNN
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity
# Bert-base
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity
# BiLSTM
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity
# LSTM
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity
# RNN
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity
# BiRNN
## ag_news
## amazon_review_full
## amazon_review_polarity
## dbpedia
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity -->