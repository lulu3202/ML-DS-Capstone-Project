# Machine Learning and Data Science Capstone Project - NASA NEO Dataset 

## Introduction
There are many dangerous bodies in space and one of them is N.E.O. - "Nearest Earth Objects". A few of these pose a danger to the planet Earth, and NASA classifies them as "is_hazardous". This dataset contains ALL NASA observations of similar objects between 1910 and 2024. 


Based on a set of input parameters,  the goal is to predict the "is_hazardous" label as accurately as possible using machine learning techniques
	- We have clearly labelled data with a clear target variable of classification – so we use Supervised learning 
	- The required output is a binary outcome of Yes or no – so we use Classification

## Dataset
- Is called nearest-earth-objects(1910-2024)  and has a total of 338199 rows × 9 columns

The dataset contains the following features:
neo_id: A unique identifier for each N.E.O.
name: The name of the N.E.O.
absolute_magnitude: The absolute magnitude of the object.
estimated_diameter_min: The estimated minimum diameter of the object.
estimated_diameter_max: The estimated maximum diameter of the object.
orbiting_body: The celestial body that the object is orbiting.
relative_velocity: The relative velocity of the object.
miss_distance: The closest distance the object will pass by Earth.
is_hazardous: A categorical label indicating whether the object is classified as hazardous (True) or not (False).


-As part of data preprocessing, I removed the null values and converted is hazardous column type from boolean to int. I have also applied standard scaler method to better optimize the input dataset.  

## Model Creation/Learning
- Data Collection
- Data Preprocessing
- Input/Output Split
- Split Train and Test Data 
- Create model with Train Data via classifier.fit method
- Evaluate/Test model performance with the Test set
- Save the best model 

## Model deployment
1 - Creating the model (in jupyter notebook)
2 - Saving the model (using pickle, save the finalized model and perform pickle dump which will result in a .sav file) 
3 - Deploying the model (create new deployment notebook to call on and open the saved model) 

## Evaluation Metric Results
	- I ran multiple classification algorithms, out of which Random Forest Classification was the best 
	- Accuracy is 0.91 - - This indicates that the model correctly classified 91% of the instances in the test dataset
