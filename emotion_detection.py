from flask import Flask, render_template, Response, jsonify
import cv2 # type: ignore
from deepface import DeepFace
import requests

app = Flask(__name__)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_emotion = "neutral"
previous_emotion = None  # Track the previous emotion

def generate_frames():
    global current_emotion
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            current_emotion = result[0]['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    global current_emotion
    return jsonify(emotion=current_emotion)

@app.route('/get_music_recommendation')
def get_music_recommendation():
    global current_emotion
    recommendations = {
        "happy": ["Happy by Pharrell Williams", "Uptown Funk by Bruno Mars"],
        "sad": ["Someone Like You by Adele", "Fix You by Coldplay"],
        "neutral": ["Shape of You by Ed Sheeran", "Blinding Lights by The Weeknd"],
        # Add more emotions and recommendations as needed
    }
    return jsonify(recommendations=recommendations.get(current_emotion, []))

@app.route('/play_music')
def play_music():
    global current_emotion, previous_emotion  # Add previous_emotion to the global variables
    if current_emotion == previous_emotion:
        return jsonify(video_url=None)  # Return None if the emotion hasn't changed

    playlist_urls = {
        "happy": "https://www.youtube.com/watch?v=ru0K8uYEZWw&list=RDQMNKED_0Rpx4g&start_radio=1",
        "sad": "https://www.youtube.com/watch?v=8ofCZObsnOo&list=RDQMogpZnpup7zY&start_radio=1",
        "neutral": "https://www.youtube.com/watch?v=pJgoHgpsb9I&list=PLP2qAKm-AAm9hIxpLbaMOcG2428lrssd5",
        # Add more emotions and playlist URLs as needed
    }
    video_url = playlist_urls.get(current_emotion, "https://www.youtube.com/watch?v=pJgoHgpsb9I&list=PLP2qAKm-AAm9hIxpLbaMOcG2428lrssd5")
    
    previous_emotion = current_emotion  # Update the previous emotion

    return jsonify(video_url=video_url)

if __name__ == "__main__":
    app.run(debug=True)