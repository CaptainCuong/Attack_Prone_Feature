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
For each dataset, I randomly trained on 1000 sentences in the train set and randomly tested on 100 sentences in the test set. Then I randomly selected 100 sentences in the test set to attack the model.
# Character CNN
## ag_news
+============================================+
|                  Summary                   |
+============================================+
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
+============================================+
## amazon_review_full
+============================================+
|                  Summary                   |
+============================================+
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
+============================================+
## amazon_review_polarity
## dbpedia
+============================================+
|                  Summary                   |
+============================================+
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
+============================================+
## imdb
## sogou_news
## yahoo_answers
## yelp_review_full
## yelp_review_polarity
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