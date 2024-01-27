# The-EuRecog-Gesture-Recognition-App
The EuRecog gesture recognition app is a hand gesture recognition system that lets a user customize ten unique hand gesture signs with questions, then based on the gesture provided by said user in usage sends the question attached to that gesture to an LLM (Open AI's gpt-3.5-turbo-instruct) then provides an answer in both text and audio form.





# Table Of Contents
* [Installation](https://github.com/Jess607/The-EuRecog-Gesture-Recognition-App#installation)
* [About the Project](https://github.com/Jess607/The-EuRecog-Gesture-Recognition-App#about-the-project)
* [How It Works](https://github.com/Jess607/The-EuRecog-Gesture-Recognition-App#how-it-works)
* [File Description](https://github.com/Jess607/The-EuRecog-Gesture-Recognition-App#file-description)
* [Licensing And Authors](https://github.com/Jess607/The-EuRecog-Gesture-Recognition-App#licensing-and-authors)

# Installation 
The code requires:
* `python 3 and above`
* `flask`
* `mediapipe`
* `gtts` 
* `tensorflow` 
* `numpy`

# About The Project 
The advent of object detection and smart gesture recognition systems has further shown the extent of the amazing potential of artificial intelligence systems. Combining computer vision processes with natural language processing (NLP) systems would prove to be even more phenomenal than what each system could achieve individually. In this project, we leverage of a pre built hand gesture recognition model called `mp hand gesture` in conjunction with the mediapipe and tensorflow libraries. Mediapipe is used to note landmarks on palms which are used to recognize gestures being made. Tensorflow is used to load the prebuilt gesture recognition model. The model includes ten hand gestures each with its individual meaning. To make things interesting, we give users the ability to edit each gesture to questions which are then sent to a large language model(particularly OPENAI's gpt-3.5-turbo-instruct) for answers. Utilizing gTTS (google text-to-speech), audio versions of text responses are also provided to users. 
Every feature is packaged in a flask app using HTML/CSS and some javascript for its UI. 



# How It Works
When users login the app, they are met with a simple homepage containing the name of the app and a prompt to begin the experience. 

![Alt text](homepage.png)

Upon clicking begin, the user is given the opportunity to customize the gestures put forward by the model. Each gesture represents a question that could potentially be asked. There are questions provided as placeholders in case users decides to not edit.

![Alt text](gestures.png)

After customizing gestures, users may then click the start webcam button. 

![Alt text](startwebcam.png)

# File Description 
The folder contains:
* `a data folder` that contains the original dataset and that generated after creating the clustering model that served as an input for the powerbi dashboard
* `customer.ipynb` a jupyter notebook of the clustering model creation 
* `customer_segment.pbix` the powerbi dashboard created- this should be opened with powerbi desktop


# Licensing And Authors
This code was created by Jessica Ogwu under the GPL-3.0 license. Please feel free to use the resources as you deem fit.