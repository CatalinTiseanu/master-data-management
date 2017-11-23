# Master Data Management

5th place (1000$) for the Master Data Management challenge on Topcoder: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&amp;rd=16529&amp;pm=13949

This took place in August 2015.

## Task Description

The challenge was to de-duplicate a dataset containing information about health providers (people - e.g., physicians, allied health providers and organizations - e.g., health service organizations, accountable care organizations)

The format of the dataset was as follows:

* ID (Integer) - Index from 0 to n-1
* Name (String) - Provider name
* Address (String) - Provider address
* Taxonomies (String) - Comma separated list of healthcare taxonomies. 
Multiple provider taxonomies may be given for a single provider.

The dataset had around 500.000 entries.

Ground truth for the training set is provided in a .csv file as a list of duplicate pairs. Each row corresponds to the two comma seperated id numbers being a duplicate pair.

The objective metric to optimize the mean squared error of prediction.
Define: dij = 1 if the ith row is a duplicate of the jth row or 0 if the ith record is not a duplicate of the jth row

The the objective was to minimize the `Brier score = sum of (Mij - dij)^2 over all pairs (i,j) where i < j`.

## My approach

My algorithm has 4 stages:

(1) Clean the initial dataset cvs’s and generate additional feature columns, such as last name, zip code or gender.
* Used a first name database in order to infer the gender of a person from the name.

(2) Use blocking, a technique used to group the initial rows (individual providers) into buckets such that it is feasible to directly compare all rows in a given bucket.
* I computed a canonical form for each word as the minimum lexicographic anagram, eg. canonical('doctor') = 'cdoort'.
* Two rows were put in the same bucket if they had the same zipcode, address, or had a lot of words in common in the name field.

(3) Construct similarity features for each pair of rows from (2), such as similarity of their last name, distance in taxonomy between their names and so on.
* Used phonetic similarity metrics to compute name and address similarity (extensions of the simple edit distance metric)

(4) Use a supervised machine learning algorithm to train on the labeled dataset from (3) in order to be able to predict on the test set.
* Used xgboost for training the supervised model. After the initial predictions, I looked at the transitive closure of the predicted duplicates to make sure I include all duplicates (if row 2 is a duplicate of row 1000, and row 1000 is a duplicate of row 3000, then row 2 and row 3000 are duplicates as well).

## Documentation

In-depth report: https://github.com/CatalinTiseanu/master-data-management/blob/master/final/Master%20Data%20Management%20-%20report.pdf
Note: the report was written as part of requirement in order to win a prize, in August 2015. It contains some typographical
errors that I'm not proud of :D, but I lost the original word document so it's hard to fix them now.

Problem statement and dataset: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16529&pm=13949
Results: https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16529

My handle is CatalinT.

The necessary env configuration is described in `final/environment_guide.md`


