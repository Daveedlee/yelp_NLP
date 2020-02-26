# yelp_NLP
# WAIT!
![](/images/caution.jpeg)

This README is under construction! More will come in a day or two. Sorry for the inconvenience.

## Overview:
 1. Datasets have been collected
 2. EDA has been performed and multiple patterns have been identified
 3. Computationally light ML classifications have been performed (Random Forest and Bernoulli Naive Bayes)
 4. Neural Network models are applied
 5. Initial findings have been presented
 6. Further modelings with different embeddings are being considered

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

*Dataset directories are empty due to Github's restriction on the file sizes. The results are replicable when the original dataset is downloaded from the link above.*

### Notebooks
This project is comprised of 5 different notebooks. The function of each component is as follows:
1. data_loader.ipynb
  * imports data from the original JSON files and divide them by states
2. cleaning_eda.ipynb
  * performs simple cleaning and visualization
3. non-neural.ipynb
  * builds models using non-neural networks, Random Forest and Bernoulli Naive Bayes
4. neural.ipynb
  * builds a model using bidirectional neural network
5. yhelper.py
  * an aggregate of all custom functions built throughout this project


## Exploratory Data Analysis
The biggest strength of this dataset is also its greatest weakness: it's absurdly huge(for a capstone). The degree of freedom allows for an infinite number of possible analyses. This abundant pool of data is great if an analyst has a gauged target, but not so if he/she's looking for one. After a long pondering and discussions with peers, I narrowed my focus down to text analysis. Although there were many interesting patterns and findings encountered during EDA, I'll limit this iteration on the text and text only.

### Approach
As I stressed earlier, this dataset is huge and that quickly became problematic. Instead of pressing shift+enter and hope things compile, I made the following key assumptions:
* It's likely that the use of language will differ from a region to the other (by states)
* Sampling 95000 reviews from each region will be an accurate depiction of how a given region do overall
* Covering the top 8 states, in terms of review counts, will provide relevant insights to the macro trend

Adhering to the assumptions above, I was able to narrow down to approximately 0.8 million reviews. Furthermore, I decided to build 8 different models based on region, instead of building a lump.


## Non-Neural Network (Random Forest and Bernoulli Naive Bayes)

## Neural Network Modeling


## Findings

## Challenges and Self-reflection
