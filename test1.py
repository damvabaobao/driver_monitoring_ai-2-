import mediapipe as mp

print("Import successful")

mp_face_mesh = mp.solutions.face_mesh

print("Creating FaceMesh...")

face_mesh = mp_face_mesh.FaceMesh()

print("FaceMesh created successfully")