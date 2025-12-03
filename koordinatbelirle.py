import cv2
import numpy as np

# Videonu buraya gir
video_path = "C:/Users/EMİRHAN/Downloads/Yolokurs/Data/highwy.mp4"
cap = cv2.VideoCapture(video_path)

# İlk kareyi oku
ret, frame = cap.read()
if not ret:
    print("Video okunamadı!")
    exit()

# Tıklanan noktaları tutacak liste
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}],") # Konsola koordinatı yazdırır
        points.append((x, y))
        
        # Tıklanan yere nokta koy
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # Eğer 4 nokta olduysa poligonu çiz
        if len(points) >= 4:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
        cv2.imshow('Koordinat Secici', frame)

cv2.imshow('Koordinat Secici', frame)
cv2.setMouseCallback('Koordinat Secici', click_event)

print("Lütfen poligon için 4 noktaya sırayla tıklayın.")
print("Çıkmak için herhangi bir tuşa basın.")

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()