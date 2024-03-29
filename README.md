# yelp_NLP

## Overview:
 1. Datasets have been collected
 2. EDA has been performed and multiple patterns have been identified
 3. Computationally light ML classifications have been performed (Random Forest and Bernoulli Naive Bayes)
 4. Neural Network models are applied
 5. Initial findings have been presented
 6. Further modelings with different embeddings are being considered

#### 2021.05.05 
 7. Adding sentiment analysis

## Introduction
![](/images/captain.gif)

*O Yelp-tain! My Yelp-tain! (pun intended. I know it's bad)*

Being the first-generation American in my family, I was always a newcomer in town. I had to figure things out on my own and when it came to restaurants, I always had to take a leap of faith. Some were alright, and some were just awful. Then, along with iPhone 3Gs came Yelp; it without a doubt changed, at very least, my consumption habit. Not only was I able to go to restaurants for a succulent meal, but also to avoid the ones that were just notorious. Despite many quarrels surrounding Yelp thereafter, I've kept this habit ever since.

Anyhow, this fanboy couldn't hold the temptation when he saw Yelp's annual challenge dataset. Especially the content I depended on for so long: the user-written reviews.

Let's dive in.


## Datasets and Notebooks
![](/images/yelp.png)
### Datasets (https://www.yelp.com/dataset/challenge)
This project relies sorely on the dataset provided by Yelp's annual challenge. No other sources, in terms of data, were used.

*Dataset directories are empty due to Github's restriction on the file sizes. The results are replicable when the original data provided by Yelp is stored in the datasets directory.*

### Notebooks
This project is comprised of 5 different notebooks. The function of each component is as follows:
1. data_loader.ipynb - imports data from the original JSON files and divide them by states
2. cleaning_eda.ipynb - performs simple cleaning and visualization
3. non-neural.ipynb - builds models using non-neural networks, Random Forest and Bernoulli Naive Bayes
4. neural.ipynb - builds a model using bidirectional neural network
5. yhelper.py - an aggregate of all custom functions built throughout this project


## Exploratory Data Analysis
The biggest strength of this dataset is also its greatest weakness: it's absurdly huge(for a capstone). The degree of freedom allows for an infinite number of possible analyses. This abundant pool of data is great if an analyst has a gauged target, but not so if he/she's looking for one. After a long pondering and discussions with peers, I narrowed my focus down to the text analysis. Although there were many interesting patterns and findings encountered during EDA, I'll limit my scope and focus solely on the text and text only.

### Approach (Assumptions)
As I stressed earlier, this dataset is huge and that quickly became problematic. Instead of pressing shift+enter and hope things compile, I made the following assumptions:
* It's likely that the use of language will differ from a region to the other (by states)
* Sampling 95000 reviews from each region will be an accurate depiction of how a given region do overall
* Covering the top 8 states, in terms of review counts, will provide meaningful insights

Adhering to the assumptions above, I was able to narrow down to approximately 0.8 million reviews. Furthermore, I decided to build **8 different models based on region,** instead of building a gigantic lump.


### Word Cloud, Bigram, and Trigram
A word cloud looks pretty fancy; words run both horizontally and vertically. Not only that, each word has a different size, color and so forth. But what does it mean?

Fundamentally, it's just a fancy histogram. It has that 'wow' factor when presenting, but when it comes to actual analytics, it's nothing more than a less-intuitive histogram. Why am I pointing this out? So that you won't treat it like a special kind; they, histograms and word clouds, are the same.

With that out of the way, let's consider the following charts from Arizona.

#### WC
![](/images/wc_az.png)
#### Bigrams
![](/images/az_bi.png)
#### Trigrams
![](/images/az_tri.png)

What's noticeable when comparing n-grams to word cloud is that the n-grams show more common phrases and compound nouns. And they are quite illuminating. For instance, the word cloud above only gives the words that can be expected of restaurant reviews-'good,' 'got', 'time,' and on. They are pretty generic and it's hard to interpret their meanings.

When observing bigrams and trigrams, the interpretability surges-we begin to see what, when, what, and how. When did the reviewers go to the restaurant? During its happy hour(bigram). What did they like the most? Sweet potato fries(trigram). And how did they complement it? By using phrases like "great food." Super interesting, aye? It gets even more interesting when comparing this macro trend region by region.

#### Similarities, Differences
Similarities among the regions, in my humble opinion, show how we 'humans' think when making a critique. The bigrams and trigrams are often dominated with the answers concerning the five Ws(when, where, why, who, and what). Also, several food names make appearances, sweet potato fries being the decisive number one among them.

Well then, what's the difference? Although subtle, each region has a very specific feature that reviewers seem to appreciate. In Nevada, for instance, the phrase "great customer service" is the overwhelming number one. With "las vegas" being its number one bigram, it can be deduced that such trigram derives from its core metropolitan area.

#### Weird Outlier
![](/images/wi_tri.png)
I didn't want to bombard this readme page with any more histogram, but this trigram from Wisconsin is worth a look. Try to find something odd.

***Ha Long Bay?!?!?!?!?!***

I was just baffled when I first saw this, but a quick Google search revealed that Ha Long Bay is the name of the Vietnamese restaurant in Madison. It nonetheless is interesting because this is the first occasion where the name of a restaurant appears in the n-gram charts.

# Modeling

## Non-Neural Network (Random Forest and Bernoulli Naive Bayes)
Adhering to Occam's Razor, I wanted to cascade my work process from the least computationally heavy to the most. Due to my target being a multi-class, I didn't have that many algorithms to go shuffle through. I narrowed my choices down to Naive Bayes (Bernoulli) and Random Forest Classifier. Following is the result:

**Test Accuracy**

|   States   | Random Forest| Bernoulli NB |
| :--------: | :----------: | :-----------:|
| Arizona    |      50%     |      56%     |
| Nevada     |      48%     |      54%     |
| North Carolina |  52%     |      50%     |
| Ohio | 52% | 50%|
| Ontario | 48% | 43%|
| Pennsylvania | 50% | 49%|
| Quebec | 49% | 48% |
| Wisconsin | 49% | 48% |

![](https://daveedlee.github.io/decker/images/gameover.gif)

They're bottom-line terrible. But why is that? Well, they all share the same story. Let's further look at a detailed classification report from Nevada.

**Random Forest**

![](/images/rf_nevada.png)

**Bernoulli Naive Bayes**

![](/images/bnb_nevada.png)

The Random Forest classifier just skips the ratings of two and three, whereas Naive Bayes performs a little better. When looking at the distribution of the samples, it's evident that the problem is caused by the class imbalance. One way to work around this issue would be using smote. Due to its computationally heavy nature, however, I wasn't able to perform this particular step. Another way would be under-sampling, and that is more feasible (And would be added to this project page later on).


## Neural Network Modeling

Instead of just arranging numbers, I wanted to have something relatable; I asked a fellow data scientist to write a review of this imaginary restaurant "Taco del Toro" and have him rate it. Being a curious mind, he wrote the following.

*My girlfriend and I went to Taco Del Toro in Lincoln Park the other day and we were very upset about how our trip panned out.
One, Lincoln Park is not our favorite neighborhood. Two, as common
Logan Square millennial vegetarians, we expected that Taco Del Toro would have vegetarian options, but they DID NOT! This is so
upsetting in the current year! Taco Del Toro needs to expand its menu for all types of customers and not just rich
Lincoln Park people who do not care about the ANIMALS. Will not be back! But overall it was a fine dining experience*

As you can tell from its content, he wanted to give the restaurant a 1-star rating. This review was tokenized using Keras tokenizer and the result from each model is as follows:

|States| Test Accuracy | Predicted Score |
| :--------: | :----------: | :-----------:|
| Arizona    |      67%    |      1     |
| Nevada     |      64%     |      5     |
| North Carolina |  62%     |      1     |
| Ohio | 61% | 5 |
| Ontario | 57% | 5|
| Pennsylvania | 60% | 4 |
| Quebec | 60% | 5 |
| Wisconsin | 59% | 5 |

only 2 models got it right :(

## Findings
#### The Problem of Two, Three, and Four
As demonstrated by the classification report from non-neural machine learning algorithms, there is a clear problem in determining the ratings of two, three, and four. Why is that? I would like to think that this isn't necessarily the problem on algorithms' part, but rather from what we put in the algorithm. We're pretty good at saying rather a restaurant is good or bad. But how good are we when describing how good or bad a restaurant is?

We're just awful. It isn't because we're less literary or dull to the sensations. We just don't have uniform criteria when writing restaurant reviews. For some the service may be the number one criteria; for others, it might be the food. Or better yet, it could be 'that' dessert. We just don't know what it means when a one says "oh this restaurant is three stars." Even better, what would happen when we bin the rating into two categories? The result is stunning: the accuracy goes up to **90 percent.**

#### GloVe vs Tokenizer
The GloVe is without a doubt great resource when vectorizing words; it can sometimes reveal the things that number obsessed data scientists could forgo inadvertently. In this case, however, the GloVe vectorization did not make much difference. The ending result using GloVe on Arizona model only increased its test accuracy by **one percent.**

#### Conclusion
![](https://daveedlee.github.io/decker/images/dance.gif)

The usual 'you get what you put in' applies to this story. The models were trained on data that doesn't know how to distinguish ratings from two to four. As a result, the ending products suffered. Perhaps it may be wise to use the broadest unit possible when classifying using vectorized sentences.
