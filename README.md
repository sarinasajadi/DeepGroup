DeepGroup: Representation Learning for Group Recommendation with Implicit Feedback:


# Introduction
DeepGroup is a group recommender system with implicit feedback. DeepGroup makes recommendation to a group of users whose personal preferences are unknown to the system and only the top choice of some of these groups are accessible. It employs a neural network to address the group recommendation problems including top choice recommendation to the group of people, and predicting the top choice of group members. 

# Environment Requirement
- tensorflow_gpu == 1.14.0
- numpy == 1.16.6

# Datasets 
- Sushi
- Irish Dublin North
- Irish Dublin West
- Irish Meath

# instructions
-For implementing the group decision prediction, set "problem_type" to "group". In order to implement reverese social choice problem (predicting group members' top choices), set "problem_type" to "user".
- We consider two types of datasets for our model and propose them with two parameter: k-participation and similarity threshold. 
- "k-participation" parameter indicates how many times a user has appeared in the groups (overlapping parameter). For the proposed datasets, it could be fixed to any of these values: 1, 2, 3, 5, 10, and 20. If you want to use similarity parameter, you should set k-participation to zero.
- "similarity_param" indicates to what extent users in a group have similar prferences. When it is equal to 0 it means that groups are formed randomly. 0.25 indicates that the kendall tau of user-user prefernces of all the group members is less than 0.25 (dissimilar groups). Also for similar groups we have thrshold of 0.75 which indicates that the kendall tau of user-user prefernces of all the group members is more than 0.75. In this type of datasets, you can also change number of groups. any value of 50, 100, 200, 500, 1000, 2000, 5000, and 10000 is supported.
