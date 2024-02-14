# The ultimate guide to AB testing

---
- [The ultimate guide to AB testing](#the-ultimate-guide-to-ab-testing)
  - [- Things to include in my write up](#--things-to-include-in-my-write-up)
  - [Reviewing articles on AB testing](#reviewing-articles-on-ab-testing)
      - [Article 1 - https://towardsdatascience.com/exploring-bayesian-a-b-testing-with-simulations-7500b4fc55bc (Blake Arnold)](#article-1---httpstowardsdatasciencecomexploring-bayesian-a-b-testing-with-simulations-7500b4fc55bc-blake-arnold)
      - [Article 2 - https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5 (Michael Frasco)](#article-2---httpsmediumcomconvoy-techthe-power-of-bayesian-a-b-testing-f859d2219d5-michael-frasco)
      - [Article 3 - http://varianceexplained.org/r/bayesian-ab-testing/ (David Robinson)](#article-3---httpvarianceexplainedorgrbayesian-ab-testing-david-robinson)
      - [Article 4 - https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html (Chris Stucchio)](#article-4---httpswwwchrisstucchiocomblog2014bayesian_ab_decision_rulehtml-chris-stucchio)
      - [Article 5 - https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf (Chris Stucchio)](#article-5---httpscdn2hubspotnethubfs310840vwo_smartstats_technical_whitepaperpdf-chris-stucchio)
      - [Article 6 - https://www.evanmiller.org/bayesian-ab-testing.html (Evan Miller)](#article-6---httpswwwevanmillerorgbayesian-ab-testinghtml-evan-miller)
      - [Article 7 - https://www.evanmiller.org/how-not-to-run-an-ab-test.html (Evan Miller)](#article-7---httpswwwevanmillerorghow-not-to-run-an-ab-testhtml-evan-miller)
      - [Article 8 - https://making.lyst.com/2014/05/10/bayesian-ab-testing/ (Maciej Kula)](#article-8---httpsmakinglystcom20140510bayesian-ab-testing-maciej-kula)
      - [Article 9 - https://www.dynamicyield.com/course/testing-and-optimization/ (Dynamic Yield)](#article-9---httpswwwdynamicyieldcomcoursetesting-and-optimization-dynamic-yield)
        - [1 - A primer on A/B testing and experimentation](#1---a-primer-on-ab-testing-and-experimentation)
        - [2 - A/A testing and decision making in experimentation](#2---aa-testing-and-decision-making-in-experimentation)
        - [3 - Why reaching and protecting statistical significance is so important in A/B tests](#3---why-reaching-and-protecting-statistical-significance-is-so-important-in-ab-tests)
        - [4 - Choosing the right traffic allocation in A/B testing](#4---choosing-the-right-traffic-allocation-in-ab-testing)
        - [5 - Understanding conversion attribution scoping in A/B testing](#5---understanding-conversion-attribution-scoping-in-ab-testing)
        - [6 - Choosing the Right Conversion Optimization Objective](#6---choosing-the-right-conversion-optimization-objective)
        - [7 - Frequentist vs. Bayesian approach in A/B testing](#7---frequentist-vs-bayesian-approach-in-ab-testing)
        - [8 - Guidelines for running effective Bayesian A/B tests](#8---guidelines-for-running-effective-bayesian-ab-tests)
      - [Article 10 - https://www.nber.org/system/files/working_papers/w15701/w15701.pdf (John List et al)](#article-10---httpswwwnberorgsystemfilesworking_papersw15701w15701pdf-john-list-et-al)
      - [Article 11 - https://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/ (Jake VDP)](#article-11---httpsjakevdpgithubioblog20140311frequentism-and-bayesianism-a-practical-intro-jake-vdp)
      - [Article 12 - http://www.qubit.com/wp-content/uploads/2017/12/qubit-research-ab-test-results-are-illusory.pdf](#article-12---httpwwwqubitcomwp-contentuploads201712qubit-research-ab-test-results-are-illusorypdf)
      - [Article 13 - https://mobiledevmemo.com/its-time-to-abandon-a-b-testing/](#article-13---httpsmobiledevmemocomits-time-to-abandon-a-b-testing)
      - [Article 14 - https://link.springer.com/article/10.3758/s13423-016-1221-4](#article-14---httpslinkspringercomarticle103758s13423-016-1221-4)
      - [Article 15 -  http://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html](#article-15----httpelemcombtillyab-testing-multiple-lookspart1-rigoroushtml)
      - [Article 16 - https://hookedondata.org/guidelines-for-ab-testing/](#article-16---httpshookedondataorgguidelines-for-ab-testing)
      - [Article 17 - http://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html (Ben Tilly)](#article-17---httpelemcombtillyab-testing-multiple-lookspart1-rigoroushtml-ben-tilly)
    - [Article 18 - https://rugg2.github.io/AB%20testing%20-%20a%20simple%20explanation%20of%20what%20power%20analysis%20does.html](#article-18---httpsrugg2githubioab20testing20-20a20simple20explanation20of20what20power20analysis20doeshtml)
    - [Article 19 - https://www.dynamicyield.com/lesson/introduction-to-ab-testing/ (dynamic yield)](#article-19---httpswwwdynamicyieldcomlessonintroduction-to-ab-testing-dynamic-yield)
    - [Article 20 - https://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html (Ben Tilly)](#article-20---httpselemcombtillyab-testing-multiple-lookspart1-rigoroushtml-ben-tilly)
  - [Things to include in my write up](#things-to-include-in-my-write-up)
---

## Reviewing articles on AB testing

#### Article 1 - https://towardsdatascience.com/exploring-bayesian-a-b-testing-with-simulations-7500b4fc55bc (Blake Arnold)

**TLDR**
 - Explains Frequentist and Bayesian (especially loss function)
 - Simulations explore relationship between true positives, threshold and sample size (i.e. test duration) in bayesian setting
 - Concludes Bayesian allows for faster testing by looking at magnitude of change as well as probability

**Summary**
 - Gives an overview of frequentist test
   - Set hypothesis
   - Determine sample size using power analysis
   - Gather data
   - Calculate the probability of observing a result at least as extreme as the data under the null hypothesis (p-value). Reject the null hypothesis and deploy the new variant to production if p-value < 5%
 - Gives an overview of Bayesian test
   - > "While the frequentist approach treats the population parameter for each variant as an (unknown) constant, the Bayesian approach models each parameter as a random variable with some probability distribution. "
   - Using Bayes rule to update a prior (often beta for sample proportion/conversion)
 - Gives overview of Bayesian decision making
   - Defines the loss of stopping the test
   - If the variant chosen was better it is 0, if it is worse then it is the difference between the metric values. Where $\theta$ is the conversion rate of each variant and $v \in \{v_1,v_2\}$ is the variant chosen
   $$
    L(\theta_{v_1},\theta_{v_2},v) = \begin{cases}
      max(\theta_{v_2}-\theta_{v_1},0) & \text{if } v=v_1\\
      max(\theta_{v_1}-\theta_{v_2},0) & \text{if } v=v_2
    \end{cases}
   $$
   - we calculate the expect loss by integrating over the posterior distributions. This takes into account the magnitude and probability of potential wrong decisions
   $$
    E[L(\theta_{v_1},\theta_{v_2},v)] = 
    \int_0^1\int_0^1
    {L(\theta_{v_1},\theta_{v_2},v)}
    f(\theta_{v_1},\theta_{v_2})
    d\theta_{v_1}d\theta_{v_2}
   $$
 - One then sets a loss threshold and stops the test when the threshold is reached
 - Monte carlo simulations
   - Example 1 in frequentist setting with the true $\bar{\theta}$ is 0.2% with expected uplift of 25% to achieve power of 80% and stat sig of 5% we would need 220k observations.
   - In the bayesian setting they assume a loss threshold of 0.004% (i.e. 2% of the initial 0.2% conversion)
   - They ran simulations (assuming the 25% improvement) to see how long the variant loss took to reach the loss threshold. In 500 simulations, they correctly chose variant B almost 90%. Moreover, 75% of the experiments concluded within 50k observations. Improving on the power and the duration!
   - Further they ran scenarios varying the observed effect and the loss threshold noting the lower the threshold the longer the test especially for smaller observed effects. They also measured the proportion of successful tests again this number decreases as the loss threshold increase or as the observed affect decreases


**Citations**
 - https://making.lyst.com/2014/05/10/bayesian-ab-testing/ (Article 8 - by Maciej Kula)
 - https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5 (Article 2 - Michael Frasco)
 - https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf (Article 5 - Chris Stucchio)


---

#### Article 2 - https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5 (Michael Frasco)

**TLDR**
 - Explains Frequentist and Bayesian (especially loss function)
 - Concludes Bayesian allows for faster testing by looking at magnitude of change as well as probability
 - Simulations show expected loss matches observed loss (over many tests)

**Summary**
 - Believes that the frequentist methodology of experimentation isn’t ideal for product innovation
 - In experiments where the improvement of the new variant is small, Bayesian methodology is more willing to accept the new variant
 - Bayesian A/B testing controls the magnitude of our bad decisions instead of the false positive rate.
 - Summary of frequentist method
   - p values and hypothesis calling test after fixed time 
 - Summary of Bayesian method
   - In Bayesian A/B testing, we model the metric for each variant as a random variable with some probability distribution then update the prior based on data
 - Issue with frequentist method
   - Unnecessarily favoring the null hypothesis (that the variant and control perform the same). In the scenario where the control is up but not quite at significance
   - Bayesian A/B testing focuses on the average magnitude of wrong decisions over the course of many experiments rather than the average number of false positives. He suggests it is better to have a few small false positives knowing that long term the average increase will be positive
 - Bayesian methodology
   - Defines the loss function 
   - Simulates over multiple runs to show long term what the loss threshold guarantees (that the loss is roughly equal to the threshold)
 - Using a prior speeds up time to conclusion. Use a slightly weaker prior than historical data suggests


**Citations**
 - https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf (Article 5 - Chris Stucchio)
 - https://www.evanmiller.org/bayesian-ab-testing.html (Article 6 - Evan Miller)
 - http://varianceexplained.org/r/bayesian-ab-testing/ (Article 3 - David Robinson)
 - https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions

---

#### Article 3 - http://varianceexplained.org/r/bayesian-ab-testing/ (David Robinson)

**TLDR**
 - The speed increase of Bayesian methods is overstated, or at least oversimplified in many of the other sources
 - Early stopping does increase the false positive rate (but as Chris comments below the promise about loss is held)

**Summary**
> Is Bayesian A/B Testing Immune to Peeking? Not Exactly

 - Freqentist peaking inflates alpha
 - Fixing this with set sample size can be frustrating in business context. You want to release a test that looks significantly up
 - The speed increase of Bayesian methods is overstated, or at least oversimplified in many of the other sources
 - > Just like frequentist methods, peeking makes it more likely you’ll falsely stop a test. The Bayesian approach is, rather, more careful than the frequentist approach about what promises it makes.
 - Nice simulations of frequentist tests being stopped early inflating alpha from 5% to 20%
 - Nice explanation of expected loss as a balance of probability and magnitude
 - Simulations of bayesian show stopping early increase false positives because:
   - Bayesian methods don’t claim to control type I error rate. Instead they set a goal about the expected loss
 - Certainly the expected loss has a relevant business interpretation (“don’t risk decreasing our clickthrough rate”)


**Citations**
 - https://www.evanmiller.org/how-not-to-run-an-ab-test.html (Article 7 - Evan Miller)
 - https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html (Article 4 - Chris Stucchio)
 - http://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html (Article 17 - Ben Tilly)
 - https://docs.swrve.com/faqs/ab-testing-faq/bayesian-approach-to-ab-testing/
 - https://warwick.ac.uk/fac/sci/psych/people/thills/thills/2013sanbornhills.pdf
 - https://www.auduno.com/2014/12/25/rapid-a-b-testing-with-sequential-analysis/
 - http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html
 - https://statmodeling.stat.columbia.edu/2014/02/13/stopping-rules-bayesian-analysis/
 - https://simplystatistics.org/2014/02/14/on-the-scalability-of-statistical-procedures-why-the-p-value-bashers-just-dont-get-it/
 - Papers:
   - When decision heuristics and science collide. Yu, E. C., Sprenger, A. M., Thomas, R. P., & Dougherty, M. R. (2013). Psychonomic Bulletin & Review, 21(2), 268–282. http://doi.org/10.3758/s13423-013-0495-z
   - Persistent Experimenters, Stopping Rules, and Statistical Inference. Steele, K. (2013). Erkenntnis, 78(4), 937–961. http://doi.org/10.1007/s10670-012-9388-1
   - The frequentist implications of optional stopping on Bayesian hypothesis tests. Sanborn, A. N., & Hills, T. T. (2014). Psychonomic Bulletin & Review, 21(2), 283–300. http://doi.org/10.3758/s13423-013-0518-9
   - Optional stopping: No problem for Bayesians. Rouder, J. N. (2014). Psychonomic Bulletin & Review, 21(2), 301–308. http://doi.org/10.3758/s13423-014-0595-4
   - Reply to Rouder (2014): Good frequentist properties raise confidence. Sanborn, A. N., Hills, T. T., Dougherty, M. R., Thomas, R. P., Yu, E. C., & Sprenger, A. M. (2014). Psychonomic Bulletin & Review, 21(2), 309–311. http://doi.org/10.3758/s13423-014-0607-4

---

#### Article 4 - https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html (Chris Stucchio)

**TLDR**
 - Outlines how to define loss and run bayesian test
 - Provides closed formula for expected loss

**Summary**

 - Defines the process of running a bayesian test
   1. Choose a "threshold of caring" - if A and B differ by less than this threshold, you don't care which one you choose.
   2. Choose a prior on the distribution of conversion rates of A and B.
   3. Compute a posterior, estimate whether the expected losses you'd make by choosing A (or B) are below the threshold of caring. If so, stop the test.
 - Advantages:
   - you can stop the test early if there is a clear winner or run it for longer if you need more samples
   - your outputs are easily interpreted quantities

 - Formula for expected loss (using Evan millers formula for the probability X > Y for two betas)

If we assume $\theta_{v_1} \sim Beta(a,b)$ and $\theta_{v_2} \sim Beta(c,d)$ then we can say

$$
\begin{aligned}
E[L(\theta_{v_1},\theta_{v_2},v_{v_2})]
&= E[L(a,b,c,d)] \\
&= \int_0^1\int_0^1
L(\theta_{v_1},\theta_{v_2},{v_2})
f(\theta_{v_1},\theta_{v_2})
d\theta_{v_1}d\theta_{v_2}\\
&= \int_0^1\int_0^1
max(\theta_{v_1}-\theta_{v_2},0)
f(\theta_{v_1},\theta_{v_2})
d\theta_{v_1}d\theta_{v_2}\\
&= \int_0^1\int_0^1
max(\theta_{v_1}-\theta_{v_2},0)
\frac{\theta_{v_1}^a(1-\theta_{v_1})^b}{B(a,b)}
\frac{\theta_{v_2}^c(1-\theta_{v_2})^d}{B(c,d)}
d\theta_{v_1}d\theta_{v_2}\\
&= \int_0^1\int_{\theta_{v_2}}^1
(\theta_{v_1}-\theta_{v_2})
\frac{\theta_{v_1}^a(1-\theta_{v_1})^b}{B(a,b)}
\frac{\theta_{v_2}^c(1-\theta_{v_2})^d}{B(c,d)}
d\theta_{v_1}d\theta_{v_2}\\
&= \frac{B(a+1,b)}{B(a,b)}h(a+1,b,c,d) -
\frac{B(c+1,d)}{B(c,d)}h(a,b,c+1,d)
\end{aligned}
$$
where
$$
h(a,b,c,d) = P(\theta_{v_1} > \theta_{v_2})
= 1- \sum_{j=0}^{c-1}
\frac{B(a+j,b+d)}{(d+j)B(1+j,d)B(a,b)}
$$


**Citations**
 - https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf (Article 5 - Chris Stucchio)
 - https://www.evanmiller.org/bayesian-ab-testing.html (Article 6 - Evan miller)


---

#### Article 5 - https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf (Chris Stucchio)

(⚠️ Warning - Long one to read)

**TLDR**
 - Overview of how to run a test.

**Summary**
 - 1 - Outlines Frequentist test hypothesis and p values
 - 3.1 - Outlines what a credible interval of a distribution is
 - 4.1 - Defines conditional probability
 - Outlines Bayes' update rule for data
 - 4.1 - Defies the prior and how comments how it is subjective
 - 4.3.1 - Shows Bayes' update with one failure and uniform prior
 - 5.1 - Defines the Beta Dist
 - Shows beta Bayes' update
 - 5.1 - Shows how to define Beta with certain mean abd variance
 - 6 - Defines the joint posterior (independent so product)
 - Show images of joint posterior for click through rates (ctr)
 - 6.1 - Defines probability of making a mistake (think of this as the total density in one triangle of his density plots)
 - INTERESTING - explains why probability of making a mistake is flawed - it considers all errors as equally bad
 - 6.2 - introduces the loss function. This is the absolute difference between the conversion rate in the variant and control if the wrong variant is chosen else 0.
 - 6.3 - defines the expected loss integrating the loss function over the joint posterior
 - 7 - defines the error tolerance
 - 7.2 - describes how to run a test, recalculating the expected loss until it is below the threshold
 - 8.1 - defines chance to beat control and chance to beat all
 - 8.3 - the sign is wrong
 - 8.3 - Calculates how many samples are required in monte carlo methods
 - 9 - how long the test takes if the two conversion rates are the same (approximates using normal)

 - Stopped at page 20 (revenue can be studied another day...)

**Citations**
 - https://www.chrisstucchio.com/blog/2015/ab_testing_segments_and_goals.html

---

#### Article 6 - https://www.evanmiller.org/bayesian-ab-testing.html (Evan Miller)

**TLDR**
 - Derives formulas for $P(A>B)$ where $A,B$ are both beta distributions
 - Similarly for $P(A>\max(B,C))$ where $A,B,C$ are  beta distributions
 - Similarly for $P(A>\max(B,C,D))$ where $A,B,C,D$ are  beta distributions
 - Also derives $P(\lambda_A>\lambda_B)$ for count data where $\lambda$ is the poisson parameter

**Citations**
 - https://www.evanmiller.org/how-not-to-run-an-ab-test.html (Article 7 - Evan Miller)

---

#### Article 7 - https://www.evanmiller.org/how-not-to-run-an-ab-test.html (Evan Miller)

**TLDR**
 - Peaking at frequentist tests dramatically inflates $\alpha$

**Summary**
 - Walks through the different scenarios that can happen when stopping an AB test early
 - For a test with a 50% conversion rate,stopping as soon as there is 5% significance, if there is no real uplift there is 26% chance you will declare a winner whilst peaking
 - > Repeated significance testing always increases the rate of false positives
 - This can be overcome by using a fixed sample size and waiting until the test reaches this size.
 - Proposes the following formula as a stopping size:
  $$
  n=16\frac{\sigma^2}{\delta^2}
  $$
 - suggests sequential experiment design and Bayesian experiment design

**Citations**
 - https://www.evanmiller.org/bayesian-ab-testing.html (Article 6 - Evan miller)
 - https://www.evanmiller.org/sequential-ab-testing.html

---

#### Article 8 - https://making.lyst.com/2014/05/10/bayesian-ab-testing/ (Maciej Kula)

**TLDR**
 - Overview of AB testing promoting Bayesian as more intuitive

**Summary**
 - Overview of AB testing
 - Overview of Bayesian prior and updating prior based on data
 - Overview of Frequentist significance testing
 - Highlight Bayesian as better for two reasons
   - repeated testing (alpha inflation)
   - low base rate problem
 - Simulating difference of sample means Vs posterior means
 - Comments that the benefits of Bayesian require a good prior


**Citations**
 - http://www.qubit.com/wp-content/uploads/2017/12/qubit-research-ab-test-results-are-illusory.pdf
 - pymc http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/

---

#### Article 9 - https://www.dynamicyield.com/course/testing-and-optimization/ (Dynamic Yield)

**TLDR**
 - The Dynamic Yield has a series on testing and optimisation

##### 1 - A primer on A/B testing and experimentation
https://www.dynamicyield.com/lesson/introduction-to-ab-testing/

**TLDR**
 - What is an A/B test?
 - Constructing a hypothesis
 - Multivariate testing and AA testing


##### 2 - A/A testing and decision making in experimentation
https://www.dynamicyield.com/lesson/aa-testing-in-experimentation/

**TLDR**
 - Bayesian Probability to be best
 - AB tests require representative samples
 - AA tests check your system works
   - Picks up bugs
   - Checks random traffic split ~50%
 - When running an AA test set the MDE (minimal detectable effect) based on what is important to detect


##### 3 - Why reaching and protecting statistical significance is so important in A/B tests
https://www.dynamicyield.com/lesson/statistical-significance/

**TLDR** 
 - Reaching proper statistical significance is critical for reliable results 
 - Explanation of hypothesis testing
 - Type I error (false positive) - We declare the variant a winner (but it wasn't really!)
 - Type II error (false negative) - We don't declare the variant a winner (but it was really)
 - Set proper sample size!
 - Statistical significance controls for type I error.
 - Statistical power/mde controls for type II error
 - Early stopping issues


##### 4 - Choosing the right traffic allocation in A/B testing
https://www.dynamicyield.com/lesson/traffic-allocation/

**TLDR** 
 - manual allocation (e.g. equal split)
 - Multi armed bandit/dynamic allocation
   - the highest-performing variation is gradually served to a larger percentage of visitors as more data is collected
   - Use this when their is limited time to collect the data e.g. hero promotion banner test


##### 5 - Understanding conversion attribution scoping in A/B testing
https://www.dynamicyield.com/lesson/traffic-allocation/

**TLDR**
 - Attribution Scoping can be at the Session-Level or at the User-Level 

##### 6 - Choosing the Right Conversion Optimization Objective
https://www.dynamicyield.com/lesson/optimization-objective/

**TLDR** 
 - Generally speaking, conversions are measured when a visitor executes actions that are defined as valuable to your business.
 - Revenue tests take longer than click/goal based tests
 - Don't peak
 - Every test requires considerable resources (budget, time, and people), so stick to the elements you know make a difference, such as messaging, hero banners, and the call-to-action (CTA).

##### 7 - Frequentist vs. Bayesian approach in A/B testing
https://www.dynamicyield.com/lesson/bayesian-testing/

**TLDR** 
 - Highlighting key differences between Bayesian and Frequentist
 - Practical implications without delving into the hard-core math
 - Promotes Bayesian as more intuitive

**Summary**
 - What is hypothesis testing?
   - variations
   - sample size
   - no peaking
   - statistical significance
 - P value misinterpretation - it is not a probability of B being better than A
 - Overview of how Bayesian methodology thinks of conversion as a distribution

**Citations**
 - https://www.evanmiller.org/how-not-to-run-an-ab-test.html (Article 7 - Evan miller)
 - http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/ (Article 11 - Jake VDP)
 - http://varianceexplained.org/r/bayesian-ab-testing/ (Article 3 - David Robinson)

##### 8 - Guidelines for running effective Bayesian A/B tests
https://www.dynamicyield.com/lesson/running-effective-bayesian-ab-tests/

**TLDR** 
 - Overview of Bayesian statistics

---

#### Article 10 - https://www.nber.org/system/files/working_papers/w15701/w15701.pdf (John List et al)

**TLDR**
**Summary**
 - "This study provides several simple rules of thumb that researchers can apply to improve the efficiency of their experimental designs"
 - Randomisation techniques
   - block and within subject
   - Factorial
<!-- Page 6 -->


**Citations**


---

#### Article 11 - https://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/ (Jake VDP)

**TLDR**
 - 
**Summary**

- Part I - Frequentism and Bayesianism: A Practical Introduction
  - Outlines the ideas of frequentist and bayesian
  - 

**Citations**

---

#### Article 12 - http://www.qubit.com/wp-content/uploads/2017/12/qubit-research-ab-test-results-are-illusory.pdf

**TLDR**
**Summary**
**Citations**

---

#### Article 13 - https://mobiledevmemo.com/its-time-to-abandon-a-b-testing/

**TLDR**
**Summary**
**Citations**


---

#### Article 14 - https://link.springer.com/article/10.3758/s13423-016-1221-4
(⚠️ Warning - Long one to read)

**TLDR**
**Summary**
**Citations**

---

#### Article 15 -  http://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html

**TLDR**
**Summary**
**Citations**

---

#### Article 16 - https://hookedondata.org/guidelines-for-ab-testing/

**TLDR**
**Summary**
**Citations**

---

#### Article 17 - http://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html (Ben Tilly)

**TLDR**
**Summary**
**Citations**

### Article 18 - https://rugg2.github.io/AB%20testing%20-%20a%20simple%20explanation%20of%20what%20power%20analysis%20does.html

**TLDR**
**Summary**
**Citations**

### Article 19 - https://www.dynamicyield.com/lesson/introduction-to-ab-testing/ (dynamic yield)

**TLDR**
**Summary**
**Citations**


### Article 20 - https://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html (Ben Tilly)

**TLDR**
**Summary**
**Citations**



---


End of article reviews

## Things to include in my write up

Things to cover in my write up
 - How do we model theta in an AB test in frequentist Vs bayesian
 - What is the central limit theorem and how does it help us
 - Frequentist test overview
   - Defining a hypothesis (one Vs two tail)
   - What is the p value
   - Power analysis to define sample size, touch on sequential testing and alpha spending maybe??
 - Bayesian overview
   - Definition of prior and posterior
   - what is the loss function (plot the value)
   - [extension] closed form of the loss function using Evan millers closed form of the chance one beta is bigger than another. Derive both Evan and Chris' equations in full.
 - Issues with peaking and alpha inflation
 - frequentist power analysis derivation and visuals -> check this https://rugg2.github.io/AB%20testing%20-%20a%20simple%20explanation%20of%20what%20power%20analysis%20does.html
 - Simulations of expected increase, sample size and power for both frequentist and bayesian. What are the formulas for frequentist - can I define for bayesian
 - Simulations regarding using a prior (too big and too small, uncertain Vs very certain, wrong prior vs right prior)

Ideas:
Frequentist Vs Bayesian philosophical. In a frequentist setting the population conversion rate is considered a fixed value and any talk of probability refers to the probability of repeated measurements against this population e.g. the probability of seeing a particular conversion rate in n users. In bayesian the conversion rate is itself considered a random variable.


**Errors**
 - Article 4 has (y−x) instead of (x-y) about half way down
 - Article 5 section 4.2 should read c_A = 0 not n_A
 - Article 5 section 4.3 figure has axis with theta - should be lambda
 - Article 5 section 8.3 - the sign is wrong should be smaller than epsilon

**Calculators**
 - https://clincalc.com/stats/samplesize.aspx
 - 