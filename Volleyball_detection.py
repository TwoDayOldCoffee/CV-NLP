import cv2 as cv2
import numpy as np
import pandas as pd

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_center = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 800:  
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2 + 1e-6)
            if 0.7 < circularity <= 1.2 and 5 < radius < 30:
                ball_center = (int(x), int(y))
                cv2.circle(frame, ball_center, int(radius), (0, 255, 0), 2)

    return frame, ball_center


def detect_players(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (100, 150), (1340, 700), 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    gray=cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)     

    haar_cascade=cv2.CascadeClassifier("haar_full_body.xml")

    faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    for (x,y,w,h) in faces_rect:  
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness=2)

    return frame, len(faces_rect) // 2, len(faces_rect) // 2 


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, ball_center = detect_ball(frame)
        frame, team1_count, team2_count = detect_players(frame)
        
        cv2.putText(frame, f'Team 1: {team1_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Team 2: {team2_count}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Volleyball Tracking', frame)
        if cv2.waitKey(25) & 0xFF == ord('d'):
            break
    cap.release()
    cv2.destroyAllWindows() 


process_video(r"C:\Users\Sahil\Documents\Summer Bootcamp\Week 2\volleyball_match.mp4")