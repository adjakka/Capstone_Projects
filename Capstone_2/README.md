A Simple Project Recommendation System with NO training
=======================================================

### Goal of the project ###

Given an amazon product description, I'm trying to see how what products will be 'recommended'. For example, if I'm purchasing a Banana Republic t-shirt, I want to see if the recommendation system just recommends other t-shirts or/and other Banana Republic products. For simplicity, I'll first visually inspect the results and then analyze results based on the descriptions. 

### Motivation for this project and practical application ###

Any decent recommender model would have been trained using data from millions of products. For companies like Amazon, such large amounts of data is easy to come by. But that's not all now is it? Can the same recommender model be used for years at a stretch without any modifications? Probably not. Products evolve over time. Companies do away with older products and bring in newer ones every day. Companies like Amazon have the time and resources to ensure that their recommendation systems are up to date with the latest trends.

Moreover, neural networks are being used for recommender systems. These models are designed using sophisticated state of the art architecture. The model needs to be designed first. Then it will be tested and modified again and again. Fine tuning deep learning models can be a monumental task. Now, what if I told you that we could come up with a decent recommender process that involves no training at all? Sounds too good to be true? Well, thanks to <small><a target="_blank" href="https://tfhub.dev/google/universal-sentence-encoder-large/3">Google's universal sentence encoder</a></small>, this is possible! Given a product description as text, the input will be mapped to a high dimensional vector of length 512. These "embeddings" are then used to compute similarities based on which practical recommendations can be made

### Explanation of some basic concepts before a deep dive ###

Please refer to my <small><a target="_blank" href="https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/notebooks/Concepts_explained.ipynb">jupyter notebok</a></small> which is part of the same project. I've tried to simplify the concepts of Transfer Learning, word embeddings and how they help us determine the semantic relationship between different sentences.

### Outline of steps ###

 1. Obtain the data from this  <small><a target="_blank" href="http://jmcauley.ucsd.edu/data/amazon/links.html">webiste</a></small>.
    I've tried this out for Amzon product files of 2 different categories: "Home and Kitchen" (Click <small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz">here</a></small> for direct link), and "Office Products" (<small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz">direct link</a></small>).
 2. Clean the data for analysis.
    Once the raw data files have been downloaded, they need to be unzipped. I was using Ubuntu 18.04. Here's how you would do it--> Navigate to the location of your raw data files. Then run the following commands:
    * gunzip meta_Home_and_Kitchen.json.gz
    * gunzip meta_Office_Products.json.gz
 3. Convert the descriptions column to a list and have that run through Tensorflow's Universal Sentence Encoder . This computes a numpy   array of length 512 for each description.
 4. Calculate the cosine similarities matrix of all the embeddings in the file.
 5. Plot the images of the products using the url in the source file.
 6. Analyze results and possibly the plot descriptions as well.
 
 - - - -
