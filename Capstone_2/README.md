A Practical Product Recommendation System with Transfer Learning
=======================================================

### Goal of the project ###
<p align="justify">
Given an amazon product description, and its sentence embeddings computed using the sentence encoder, can we compute the nearest neighbor embeddings and have the encoder work like a recommendation system? For example, if we have a Banana Republic t-shirt, using this method, can we expect to see other similar t-shirts or other Banana Republic products? That would be very useful for an e-commerce/retail company, wouldn't you agree? 
</p>

### Motivation for this project and practical application ###
<p align="justify">
Any decent recommender model would have been trained using data from millions of products. For companies like Amazon, such large amounts of data is easy to come by. But that's not all! Can the same recommender model be used for a year or months at a stretch without any modifications? Probably! Or probably not. Products evolve over time. Companies do away with older products and bring in newer ones every day. Companies like Amazon have the time and resources to ensure that their recommendation systems are up to date with the latest trends.
</p>

<p align="justify">
 It's also important to consider data scarcity, which is something a lot of companies have to deal with it. Good quality data isn't easy to come by. A machine learning model is only as good as the data it is fed. Even the best data scientists cannot work together to put together a state of the art machine learning model without data. More  data-->Better Machine Lerning models. This is one of my major motivations for this project. To counter data scarcity.
</p>


<p align="justify">
Moreover, neural networks are being used for recommender systems. These models are designed using sophisticated state of the art architecture. The model needs to be designed first. Then they will be tested and modified again and again. Fine tuning deep learning models can be a monumental task. Now, what if I told you that we could come up with a decent recommender process that involves no training at all? Sounds too good to be true? Well, thanks to <small><a target="_blank" href="https://tfhub.dev/google/universal-sentence-encoder-large/3">Google's universal sentence encoder</a></small>, this is possible! Given a product description as text, this text input will be mapped to a high dimensional vector space (512 dimensions). These "embeddings" are then used to compute similarities based on which practical recommendations can be made.
</p>

### Sneak peek of what we're getting into ###

So, what am I really talking about? Allow me to illustrate using pictures. After all, a picture is worth thousands of words!

![picture alt](https://github.com/adjakka/Miscellaneous/blob/master/Kitchen_items.JPG "Product and its related products")

<p align="justify">
What I did here was take the embedding of the descriptions (the first product here) and then, look for descriptions of products that were similar to the first product. The pictures are for illustrative purposes alone. Check out another interesting result:
 </p>
 
 ![picture alt](https://github.com/adjakka/Miscellaneous/blob/master/kithen_blender.JPG "Product and its related products")
 <p align="justify"> 
 Given a blender glass jar, we see that the related products are either blender parts or other blenders. Let's check out something from the clothing category next!
 </p>
 
 
 ![picture alt](https://github.com/adjakka/Miscellaneous/blob/master/clothes.JPG "Product and its related products")
<p align="justify"> 
 Given a Saints jacket (Saints are an NFL team), we get jackets of other NFL brands and even some earrings. This is because it is a womens jacket. I hope we can agree that this is pretty intriguing!
 
 Since I'm a movie buff, let's check out another result:
![picture alt](https://github.com/adjakka/Miscellaneous/blob/master/moviesnew.JPG "Product and its related products")
<p align="justify">
How interesting, isn't it? Given the movie plot of Guardians of the Galaxy, the module was able to find other fantasy/action/adventure/Superhero movies! Just goes to show how well trained the Universal Sentence Encoder is! We could achieve something so meaningful without even training or building a model of our own. All I did was use the model as is with the pretrained weights! 
</p>

### Explanation of some basic concepts before a deep dive ###
<p align="justify">
Please refer to my <small><a target="_blank" href="https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/notebooks/Concepts_explained.ipynb">jupyter notebok</a></small> which is part of the same project. I've tried to simplify the concepts of Transfer Learning, word embeddings and how they help us determine the semantic relationship between different sentences.
 </p>

### Requirements ###

Please refer to this [requirements file](https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/requirements.txt) to install all dependencies needed for this project.

#### Installing Tensorflow with GPU support (Optional) ####
<p align="justify">
GPUs are used to dramatically increase the speed of tasks involving numerical computations in Python. While it isn't compulsary, I highly recommend using  GPU support for speed. If you're interesred in setting up Tensorflow with GPU support, you may want to refer to this <small><a target="_blank" href="https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/comment-page-3/#comments">excellent guide for Linux users by Arun Mandal</a></small>. Mac users may want to refer to <small><a target="_blank" href="https://gist.github.com/ageitgey/819a51afa4613649bd18">this guide</a></small>
 </p>
 
### Outline of steps ###
 1. Obtain the data from this  <small><a target="_blank" href="http://jmcauley.ucsd.edu/data/amazon/links.html">webiste</a></small>.
    I've tried this out for Amzon product files of 3 different categories: "Home and Kitchen" (Click <small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz">here</a></small> for direct link), and "Office Products" (<small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz">direct link</a></small>). Clothes and jewelry json file can be directly <small><a target="_blank" href="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz"> downloaded here.</a></small>
    Lastly, the movies dataset can be downloaded <small><a target="_blank" href="https://www.kaggle.com/rounakbanik/the-movies-dataset">here.</a></small>
    
    Once downloaded, you may want to move the files into the raw data folder.
 2. Clean the data for analysis.
    Once the raw data files have been downloaded, they need to be unzipped. I was using Ubuntu 18.04. Here's how you would do it--> Navigate to the location of your raw data files. Then run the following commands (for Ubuntu users) using terminal:
    * gunzip meta_Home_and_Kitchen.json.gz
    * gunzip meta_Office_Products.json.gz
    * gunzip meta_Clothing_Shoes_and_Jewelry.json.gz
   
    (Mac users may want to refer to this [guide](https://www.dummies.com/computers/macs/how-to-zip-and-unzip-files-on-your-mac/) to unzip files)
    
    You may want to move the files to the [interim data folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/data/interim) after the above step.
   Once the files have been unzipped, we'll have the json files in place. Json files aren't easy to work with and hence, its up to us to convert the data in this json file into a nice Pandas dataframe that we can use. The cleanup files can be found in the [interim data folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/data/interim). But you may directly jump into the notebooks using the links below:  
    * [Clean up Home and kitchen json](https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/data/interim/Home_kitchen.ipynb)(for cleaning "Home and kitchen" json), 
    
    * [Clean up Office products json](https://github.com/adjakka/Springboard_Capstone_Projects/blob/master/Capstone_2/data/interim/Office_products_cleanup.ipynb)(for Office products json file). 
    
    * [Clean up clothes and jewelry json](https://github.com/adjakka/Capstone_Projects/blob/master/Capstone_2/data/interim/Clothing_data_preprocess.ipynb)(for clothes and jewelry json file). 
    
    * [Clean up movies data file](https://github.com/adjakka/Capstone_Projects/blob/master/Capstone_2/data/interim/Movie_file_cleanup.ipynb)(for movies data file). 
    
    The cleaning steps are exactly the same line for both files. They only differ by file names. Once done, you want to move the files to the [processed data folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/data/processed)
    
    **[Note: I haven't added the raw and interim data files as they're too big. However, the processed data files are in place. If you have all the required dependencies mentioned in the requirements file, then the processed files are all you'll need to get started.]**

 3. Convert the descriptions column to a list and have that run through Tensorflow's Universal Sentence Encoder . This computes a numpy   array of length 512 for each description.
 4. Calculate the cosine similarities matrix of all the embeddings in the file.
 5. Plot the images of the products using the url in the source file.
 6. Analyze results and possibly the product descriptions as well.
    Steps 3,4,5 and 6 can be followed by referring to my files in the [notebooks folder](https://github.com/adjakka/Springboard_Capstone_Projects/tree/master/Capstone_2/notebooks) **[Please look for files with the word "semantic" here]**
    
For a detailed report on this project, please refer to [this PDF.](https://github.com/adjakka/Capstone_Projects/blob/master/Capstone_2/reports/Capstone_2_analysis.pdf)
  
 - - - -
