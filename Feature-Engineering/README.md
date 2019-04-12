These are the FE codes for this competition both from the top teams and myself.<br/>

Everyone could have their own understanding of the data. Here is how I find the first FE in the last day:<br/>

1. By plotting the distribution between target 0 and target 1, I found that the big gap is always near the two sides and the top. So I want to let lgb know this, as many hints say using extra information of unique value counts could help.<br/>
2. I found that the value of the top area always has much more counts and the value of the two sides area has little unique counts, (larger probability gets more counts). So maxcounts and mincounts could be useful, where maxcounts is the maximum unique value counts of the feature column, mincounts is the minimum unique value counts of the feature column.<br/>
3. Then the question is how to tell lgb which point is near top or near two sides easier, min(abs(x-maxcounts)/(maxcounts-mincounts), abs(x-mincounts)/(maxcounts-mincounts)) could work, you can view this as a metric of distance.<br/>
4. Why not use the information from test set (more data more power), use all the dataset to get unique counts for each value.
5. If you do 1-4 with magic parameters, you can get 0.902, but how to get 0.912, finetune your parameters, this could boost to 0.912. I only tried several parameters, and maybe this is not the best result.<br/>

Since I have no experience with EDA, FE, it's hard for me to find this, but I'm lucky and thanks for all the great kernels and all the hints from all of you. Learnt a lot from this competition, second FE is logical but I have no experience, no creativity and no time to think out it. My first FE may not be the best, but it's logical for me. Keep learning and keep moving!!
