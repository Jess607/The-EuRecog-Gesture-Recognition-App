#import libraries
from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from openai import OpenAI
import time
import threading
from queue import Queue
from gtts import gTTS
from generate_frames import generate_frames



#create flask app
app = Flask(__name__)



#a variable show_text is created and set to false 
show_text = False

#answer stores response from LLM 
answer=''

    
    
#this route returns the home page 
@app.route('/')
def index():
    global answer

    text = answer if show_text else ""
    return render_template('index.html', text=text)



#this returns the video feed from the web cam for gesture recognition
@app.route('/video_feed')
def video_feed():
    #renders video frames for capturing of gestures
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



#returning text from gesture recognition process 
@app.route('/get_text')
def get_text():
    #returns answer on screen if show_text is true
    global show_text
    global answer
    
    if show_text:
        return answer
    return ""



#speech from response returned from llm
@app.route('/get_speech')
def get_speech():
    #converting text response to audio 
    global answer
    myobj = gTTS(text=answer, lang="en", slow=False)
    myobj.save("welcome.mp3")
    with open(r'C:\Users\jessica.ogwu\Documents\The-EuRecog-Gesture-Recognition-App\hand-gesture-recognition-code (1)\welcome.mp3', 'rb') as audio_file:
        audio_data = audio_file.read()
    return Response(audio_data, content_type='audio/mpeg')


#editing gesture.names file to fit inputs provided by user
@app.route('/process_gestures', methods=['POST'])
def process_form():
    #inputs from gestures entered by users
    okay = request.form.get('okay')
    peace = request.form.get('peace')
    thumbs_up = request.form.get('thumbs up')
    thumbs_down = request.form.get('thumbs down')
    call_me = request.form.get('call me')
    stop = request.form.get('stop')
    rock = request.form.get('rock')
    live_long = request.form.get('live long')
    fist= request.form.get('fist')
    smile = request.form.get('smile')
    gesture_list=[okay,peace,thumbs_up, thumbs_down, call_me, stop, rock, live_long, fist, smile]
    with open(r'C:\Users\jessica.ogwu\Documents\The-EuRecog-Gesture-Recognition-App\hand-gesture-recognition-code (1)\gesture.names', 'w') as file:
        for i in gesture_list:
            file.writelines(i +'\n')

    
    
    return render_template('third.html',q1=okay, q2=peace, q3=thumbs_up, q4=thumbs_down, q5=call_me, q6=stop, q7=rock, q8=live_long, q9= fist, q10=smile)


#renders html page with editable gesture actions
@app.route('/get_gestures')
def second():
    return render_template('second.html')



if __name__ == '__main__':
    app.run(debug=True, port=8000)