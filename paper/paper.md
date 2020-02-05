---
title: 'Autorank: A Python package for automated ranking of classifiers'
tags:
  - Python
  - statistics
  - ranking
  - automation
authors:
  - name: Steffen Herbold
    orcid: 0000-0001-9765-2803
    affiliation: 1
affiliations:
 - name: Institute for Computer Science, University of Goettingen, Germany
   index: 1
date: 15 January 2020
bibliography: paper.bib
---

# Summary

Analyses to determine differences in the central tendency, e.g., mean or median values, are an
important application of statistics. Often, such comparisons must be done with paired samples, i.e., populations that
are not dependent on each other. This is, for example, required if the performance different machine learning algorithms
should be compared on multiple data sets. The performance measures on each data set are then the paired samples, the
difference in the central tendency can be used to rank the different algorithms. This problem is not new and how such
tests could be done was already described in 2006  in the well-known article _Janez Demšar. 2006. Statistical Comparisons
of Classifiers over Multiple Data Sets. J. Mach. Learn. Res. 7 (December 2006), 1–30_. 

Regardless, the correct use of Demšar guidelines is hard for non-experts in statistics. The distribution of the
populations must be analyzed with the Shapiro-Wilk test for normality and, depending on the normality with Levene's 
test or Bartlett's tests for homogeneity of the data. Based on the results and the number of populations, 
researchers must decide
whether the paired t-test, Wilcoxon's rank sum test, repeated measures ANOVA with Tukey's HSD 
as post-hoc test, or Friedman's tests and Nemenyi's post-hoc test is the suitable statistical framework. 
All this is already quite complex. Additionally, researchers must adjust the significance level due to the number of
tests to achieve the desired family-wise significance and control the false-positive rate of the test results. 
Finally, good reporting of the results requires the calculation of confidence intervals, effect sizes and the decision
whether it is appropriate to report the mean value and standard deviation, or whether the median value and the median 
absolute deviation is better suited.   

The goal of Autorank is to simplify the statistical analysis for non-experts. Autorank takes care of all of the above
with a single function call. Additional functions allow the generation of appropriate plots, result tables, and even of
a complete latex document. All that is required is the data about the populations is in a dataframe. Therefore, we 
strongly believe that Autorank can help  


# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

# References