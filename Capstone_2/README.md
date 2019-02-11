A Simple Project Recommendation System with Transfer Learning
=======================================================

### Goal of the project ###
<p align="justify">
Given an amazon product description, I'm trying to see how what products will be 'recommended'. For example, if I'm purchasing a Banana Republic t-shirt, I want to see if the recommendation system just recommends other t-shirts or/and other Banana Republic products. Once we have the results for a certain product, I'll first visually inspect the results (the recommended products) and then analyze results based on the descriptions. 
</p>

### Motivation for this project and practical application ###
<p align="justify">
Any decent recommender model would have been trained using data from millions of products. For companies like Amazon, such large amounts of data is easy to come by. But that's not all now is it? Can the same recommender model be used for years at a stretch without any modifications? Probably not. Products evolve over time. Companies do away with older products and bring in newer ones every day. Companies like Amazon have the time and resources to ensure that their recommendation systems are up to date with the latest trends.
</p>

<p align="justify">
Moreover, neural networks are being used for recommender systems. These models are designed using sophisticated state of the art architecture. The model needs to be designed first. Then it will be tested and modified again and again. Fine tuning deep learning models can be a monumental task. Now, what if I told you that we could come up with a decent recommender process that involves no training at all? Sounds too good to be true? Well, thanks to <small><a target="_blank" href="https://tfhub.dev/google/universal-sentence-encoder-large/3">Google's universal sentence encoder</a></small>, this is possible! Given a product description as text, the input will be mapped to a high dimensional vector of length 512. These "embeddings" are then used to compute similarities based on which practical recommendations can be made.
</p>

### Explanation of some basic concepts before a deep dive ###

Please refer to my <small><a target="_blank" href="https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/notebooks/Concepts_explained.ipynb">jupyter notebok</a></small> which is part of the same project. I've tried to simplify the concepts of Transfer Learning, word embeddings and how they help us determine the semantic relationship between different sentences.

### Requirements ###

Please refer to this [requirements file](https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/requirements.txt) to install all dependencies needed for this project.

#### Installing Tensorflow-GPU (Optional) ####

GPUs are used to dramatically increase the speed of tasks involving numerical computations in Python. While it isn't compulsary, I highly recommend using  GPU support for speed. If you're interesred in setting up Tensorflow with GPU support, you may want to refer to this [excellent guide by Arun Mandal](https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/comment-page-3/#comments).

### Outline of steps ###
<p align="justify">
 1. Obtain the data from this  <small><a target="_blank" href="http://jmcauley.ucsd.edu/data/amazon/links.html">webiste</a></small>.
    I've tried this out for Amzon product files of 2 different categories: "Home and Kitchen" (Click <small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz">here</a></small> for direct link), and "Office Products" (<small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz">direct link</a></small>). Once downloaded, you may want to move the files into the raw data folder.
 </p> 
 2. Clean the data for analysis.
    Once the raw data files have been downloaded, they need to be unzipped. I was using Ubuntu 18.04. Here's how you would do it--> Navigate to the location of your raw data files. Then run the following commands (for Ubuntu users) using terminal:
    * gunzip meta_Home_and_Kitchen.json.gz
    * gunzip meta_Office_Products.json.gz
   
    (Mac users may want to refer to this [guide](https://www.dummies.com/computers/macs/how-to-zip-and-unzip-files-on-your-mac/) to unzip files)
    
    You may want to move the files to the [interim data folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/data/interim) after the above step.
   Once the files have been unzipped, we'll have the json files in place. Json files aren't easy to work with and hence, its up to us to convert the data in this json file into a nice Pandas dataframe that we can use. The cleanup files can be found in the [interim data folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/data/interim). But you may directly jump into the notebooks using the links below </p>:  
    * [Clean up Home and kitchen json](https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/data/interim/Home_kitchen.ipynb)(for cleaning "Home and kitchen" json), 
    
    * [Clean up Office products json](https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/data/interim/Office_products_cleanup.ipynb)(for Office products json file). 
    
    The cleaning steps are exactly the same line for both files. They only differ by file names. Once done, you want to move the files to the [processed data folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/data/processed)
    
    **[Note: I haven't added the raw and interim data files as they're too big. However, the processed data files are in place. If you have all the required dependencies mentioned in the requirements file, then the processed files are all you'll need to get started.]**

 3. Convert the descriptions column to a list and have that run through Tensorflow's Universal Sentence Encoder . This computes a numpy   array of length 512 for each description.
 4. Calculate the cosine similarities matrix of all the embeddings in the file.
 5. Plot the images of the products using the url in the source file.
 6. Analyze results and possibly the plot descriptions as well.
 </p> 
    Steps 3,4,5 and 6 can be followed by referring to my files in the [notebooks folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/notebooks) **[Please look for files with the word "semantic" here]**
  
 - - - -
