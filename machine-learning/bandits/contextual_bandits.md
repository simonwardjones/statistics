# A Contextual-Bandit Approach to Personalized News Article Recommendation

Paper link: https://arxiv.org/pdf/1003.0146


## Abstract

> a learning algorithm sequentially selects articles to serve
users based on contextual information about the users and articles,
while simultaneously adapting its article-selection strategy based
on user-click feedback to maximize total user clicks

## Introduction

A small amount of traffic can be designated for such exploration. Based on the users’ response (such as clicks) to randomly selected content on this small slice of traffic, the most popular content can be identified and exploited on the remaining traffic. This strategy, with random exploration on an $\epsilon$ fraction of the traffic and greedy exploitation on the rest, is known as $\epsilon$ greedy.

In this paper, we formulate it as a contextual bandit problem, a principled approach in which a learning
algorithm sequentially selects articles to serve users based on contextual information of the user and articles, while simultaneously adapting its article-selection strategy based on user-click feedback
to maximize total user clicks in the long run.

### Multi Arm bandit formulation

Formally, a contextual-bandit algorithm A proceeds in discrete trials t = 1, 2, 3, . . . In trial t:
1. The algorithm observes the current user ut and a set At of
arms or actions together with their feature vectors xt,a for
a ∈ At. The vector xt,a summarizes information of both the
user ut and arm a, and will be referred to as the context.
2. Based on observed payoffs in previous trials, A chooses an
arm at ∈ At, and receives payoff rt,at whose expectation
depends on both the user ut and the arm at.
3. The algorithm then improves its arm-selection strategy with
the new observation, (xt,at
, at, rt,at
). It is important to emphasize here that no feedback (namely, the payoff rt,a) is
observed for unchosen arms a 6= at. The consequence of
this fact is discussed in more details in the next subsection.


Resources:
 - https://proceedings.mlr.press/v15/chu11a/chu11a.pdf
 - https://truetheta.io/concepts/reinforcement-learning/lin-ucb/


Final write ups:
 - https://www.simonwardjones.co.uk/posts/thompson_sampling/
 - https://www.simonwardjones.co.uk/posts/contextual_bandits/

