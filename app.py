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
import asyncio
from gtts import gTTS
from chat import llm_result


#create flask app
app = Flask(__name__)

#creating a queue to store gesture response from image captured
out_q=Queue()


frame_lock = threading.Lock()

#a variable show_text is created and set to false 
show_text = False

#answer stores response from LLM 
answer=''





def generate_frames():
    global video_stream, frame_lock, show_text, answer
    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = tf.keras.models.load_model('mp_hand_gesture')
    print('Success')
    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)
    cap = cv2.VideoCapture(0)
    start_time=time.time()

    #capturing happens for 10 seconds
    while time.time()-start_time<10:


        ret, frame = cap.read()
        x, y, c = frame.shape

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)
        
        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])



                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 2, cv2.LINE_AA)


        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        
        frame = buffer.tobytes()
        with frame_lock:
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    show_text=True
    
    #if there is a gesture detected, gesture is added to queue
    if className!='':
        out_q.put(className)

    #gesture is retrieved from queue and passed to function that answers question
    final_c=out_q.get()
    answer=llm_result(final_c)


    #prints results on terminal
    print(type(out_q.get()))
    print(show_text)
    
    
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