Disaster Tweet Classification Through Multi-Task Learning and Bayesian Optimization <!-- omit from toc -->
==============================

- [Summary](#summary)
- [Data Sources](#data-sources)
- [Results](#results)
  - [Method 1](#method-1)
  - [Method 2](#method-2)
  - [Method 3](#method-3)
- [Team Members](#team-members)


Summary
------------
Developed and trained a RoBERTa-based multi-task model to classify tweets whether they refer to a real disaster or not. Leverages a separete tweet dataset with sentiment labels to help in the disaster classification task. This is achieved by utilizing a neural network architecture with two heads, one for disaster classification (Task 1) and one for sentiment classification (Task 2). A weighted loss, which weights and sums the losses of the two classification tasks is ultimately used during back propagation to incorporate sentiment information during training. The weights of the loss function are tuned using Bayesian Optimization which maximizes the F1 score of Task 1.

This repository also includes two alternative strategies in predicting disaster tweets but results are not reported.

Data Sources
------------
1. [Disaster Tweets Prediction Competition](https://www.kaggle.com/competitions/nlp-getting-started/overview)<br>
Labels: 1, if tweet is about a real disater, 0 otherwise
1. [Twitter Tweets Sentiment Dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)<br>
Labels: Negative, Neutral or Positive Sentiment

Results
------------
 ### Method 1 ###
 Minimizing separate loss functions for Task 1 and Task 2 during training.
| Task                        | F1 Score | Accuracy |
| :-------------------------- | -------: | -------: |
| 1. Disaster Classification  |   81.92% |   83.59% |
| 2. Sentiment Classification |   77.81% |   77.84% |

**Confusion Matrix for Task 1**

![t1_tm1](/notebooks%202/t1_tm1.png)

**Confusion Matrix for Task 2**

![t2_tm1](/notebooks%202/t2_tm1.png)


### Method 2 ###
Minimizing a combined weighted loss, $\lambda_1l_1 + \lambda_2l_2$  where $\lambda_1$ and  $\lambda_2$ are positive scalar weights, and $l_1$ and $l_2$ are the task-specific loss functions. Uses $\lambda_1=\lambda_2=0.5$

| Task                        | F1 Score | Accuracy |
| :-------------------------- | -------: | -------: |
| 1. Disaster Classification  |   81.55% |   84.64% |
| 2. Sentiment Classification |   75.81% |   75.97% |

**Confusion Matrix for Task 1**

![t1_tm2](/notebooks%202/t1_tm2.png)

**Confusion Matrix for Task 2**

![t2_tm2](/notebooks%202/t2_tm2.png)


### Method 3 ###
Same method as Method 2, but both lambdas tuned using Bayesian Optimization by optimizing the model's disaster classification F1 score (Task 1) which yielded $\lambda_1 = 0.6899408753961325$, $\lambda_2= 0.4041465884074569$. 

After tuning, inference was made on unlabeled test tweets and predictions were submitted to the [Disaster Tweets Prediction Competition](https://www.kaggle.com/competitions/nlp-getting-started/overview).

| Task                    | Test Accuracy |
| :---------------------- | ------------: |
| Disaster Classification |        83.85% |

Team Members
------------
- Nathan Casanova
- Kaushik Asok
- Pragati Sangal