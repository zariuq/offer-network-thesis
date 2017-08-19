Executive Summary:


Offer Networks are "barter exchange networks" where users upload (offer, request) pairs and receive suggested n-way exchanges to accept or reject. 
The core idea is to enhance and formalize human organization in markets of services/goods (tasks).

The thesis contains a good review of the related academic literature. 80% of the research focuses on kidney exchange, as you can't sell those for $$.

My work represents the offer network in Neo4j. 

Three forms of matching algorithms are tested:
1) MAX: A maximum edge-weight cycle-cover is found using munkres algorithm. 
2) GSC: A greedy shortest cycle-cover algorithm, quickly implemented in Neo4j's Cypher query language.
3) TWO: A maximum two-way cycle cover. It sucks balls.

MAX is optimal if users always accept suggested matches. Otherwise, shorter exchanges are better.
Even in its optimal case, GSC performs almost as well as MAX. 
Dynamic matching based only on newly added (or just rejected) (offer, request) pairs performs about as well too.
Choosing the order of GSC to prioritize unpopular tasks marginalizes them less, and in the long run, performs marginally better.


- Matching frequency (in terms of number of (offer, request) pairs, OR-pairs, added) does not seem to matter much.

- Once the graph is big enough (over 10,000 OR-pairs), wait time for matched nodes seem to stabilize, at around 1000.

- Running from 100 to 35,000 OR-pairs, there is little variance in how many OR-pairs are matched. About 1/3 of added OR-pairs get matched and the graph size slowly increases. 


As the acceptance probability decreases, the speed at which OR-pairs are matched decreases significantly.

A heuristic I called HOR (hold OR-pair) allows OR-pairs in a large exchange to get processed even if some users reject.
For example:

OR-pairs: a,b,c,d
Accept:   0,1,1,1

c can be processed. A new OR-pair representing b and d's unsatisfied (offer, request) is made. a is treated as usual.

Using HOR, performance is not only maintained as acceptance probability decreases, but the potential for matches in the Offer Network is actually depleted faster!

OR-pairs can be in multiple cycles in the graph, but normally only used in one of the exchanges. HOR allows some OR-pairs to, in essence, be part of two+ exchanges. 
Additionally, HOR saves some potential for future matching steps, which has been found to be helpful when intelligently done in kidney exchange networks.


Unfortunately, I lazily allowed rematching of OR-pairs even after one user rejects the match. I suspect the results will be more interesting when this 'feature' is removed.

HOR can be extended to a form of asymmetric exchanges resembling a formalism for gifts.


In Conclusion, matching 1/3 of added OR-pairs is promising. 

I was sceptical of the whole gift economy idea, but the research on gift-chains and my experiments here indicate that an Offer Network with even 5% gifts will be much, much more effective.


