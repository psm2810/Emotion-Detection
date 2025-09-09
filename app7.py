import streamlit as st
import cv2  #for video processing and image manipulation.
import tempfile  #create temporary files for storing uploaded videos
import os
from deepface import DeepFace  #provides tools for facial recognition and emotion analysis

def detect_faces_and_emotions(video_path, output_path):
    #Video Capture Setup
    cap = cv2.VideoCapture(video_path)  #Opens the video file specified by video_path
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  #Retrieves the frame rate of the video
    frame_skip = 1  #Sets how many frames to skip (in this case, it processes every frame)  # Only process every 10th frame to save time  

    #Video Writer Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #Defines the codec for encoding the output video
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))   # Initializes a video writer that will save the processed video to output_path with same frame rate & dimensions as the input video.
    

    #Frame Processing Loop
    current_frame = 0
    while cap.isOpened():  #Enters a loop that continues while the video capture is open.
        ret, frame = cap.read()
        if not ret:  #If reading the frame fails (not ret), the loop breaks.
            break
        

        #Face Detection and Emotion Analysis
        '''
        Checks if the current frame should be processed (based on frame_skip).
        Converts the frame to grayscale, which is necessary for face detection.
        Loads a pre-trained Haar cascade classifier for face detection and applies it to the grayscale frame.
        '''
        if current_frame % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
            
            #Drawing Rectangles and Analyzing Emotion
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_frame = frame[y:y+h, x:x+w]  #Extracts the region of interest (the detected face) into face_frame.
                try:
                    """
                    Attempts to analyze the emotion of the detected face using DeepFace's analyze method.
                    If analysis results are a list, it extracts the dominant emotion and puts it as text above the face rectangle on the frame.
                    If an error occurs during emotion detection, it catches the exception and displays an error message using Streamlit.
                    """
                    analysis_results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
                    if isinstance(analysis_results, list):
                        for analysis in analysis_results:
                            if 'dominant_emotion' in analysis:
                                emotion = analysis['dominant_emotion']
                                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    st.error(f"Error in emotion detection: {e}")
        
        out.write(frame)   #Writes the processed frame (with rectangles and emotion text) to the output video file.
        current_frame += 1  #Increments the current_frame counter.
    
    cap.release() #Releases the video capture and writer resources after processing is complete.
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
