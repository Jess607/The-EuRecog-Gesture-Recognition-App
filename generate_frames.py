#import libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import threading
from queue import Queue
from chat import llm_result


#creating a queue to store gesture response from image captured
out_q=Queue()

#handle threading
frame_lock = threading.Lock()

#Function that handles capturing of individual via webcam
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