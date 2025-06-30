import streamlit as st
import cv2
import tempfile
import os
from deepface import DeepFace

def detect_faces_and_emotions(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 1  # Only process every 10th frame to save time
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_frame = frame[y:y+h, x:x+w]
                try:
                    analysis_results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
                    if isinstance(analysis_results, list):
                        for analysis in analysis_results:
                            if 'dominant_emotion' in analysis:
                                emotion = analysis['dominant_emotion']
                                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    st.error(f"Error in emotion detection: {e}")
        
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()

def main():
    st.title("Facial Emotion Detection")
    
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        # Display input video
        st.video(tfile.name)
        
        if st.button("Detect Emotions"):
            with st.spinner('Processing...'):
                output_video_path = 'output.mp4'  # Specify a fixed output path
                detect_faces_and_emotions(tfile.name, output_video_path)
                
                # Display output video
                video_file = open(output_video_path, 'rb')
                st.video(video_file)
                video_file.close()  # Close the file after displaying the video
                
                # Cleanup
                os.remove(tfile.name)  # Remove the temporary input file
                # Optionally, you might also want to remove the output video file after it's been displayed
                # os.remove(output_video_path)

if __name__ == "__main__":
    main()
