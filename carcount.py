import numpy as np
import cv2
from ultralytics import YOLO
import time


class carcount:
    """Trafik videosunda araç türlerini sayacak sınıf"""
    
    ARAC_SINIFI = {
        2: "araba",
        3: "motorsiklet",
        5: "otobüs",
        7: "kamyon"
    }
    
    def __init__(self, video_path, model_path="yolo11n.pt"):
        """
        Sınıfı başlatır
        
        Args:
            video_path: Video dosyasının yolu
            model_path: YOLO model dosyasının yolu
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError("Dosya acilamadi")
        
        self.genislik = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.yukseklik = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.model = YOLO(model_path)
        
        # Poligon bölgeleri (sol ve sağ şerit için)
        # cv2.pointPolygonTest ile çalışmak için numpy array olarak tanımlanır
        self.sol_bolge = np.array([
            [80, 450],
            [300, 300],
            [500, 300],
            [450, 450]
        ], dtype=np.int32)
        
        self.sag_bolge = np.array([
            [580, 300],
            [self.genislik-320, 300],
            [self.genislik-100, 450],
            [630, 450]
        ], dtype=np.int32)
        
        # Araç türü istatistikleri
        self.sol_serit_istatistik = {"araba": 0, "motorsiklet": 0, "otobüs": 0, "kamyon": 0}
        self.sag_serit_istatistik = {"araba": 0, "motorsiklet": 0, "otobüs": 0, "kamyon": 0}
        
        # Serit sayaçları
        self.sol_sayac = 0
        self.sag_sayac = 0
        
        # Takip edilen araç ID'leri
        self.tracked_ids = {"sol": set(), "sag": set()}
        
        # Frame atlama parametreleri
        self.frame_skip = 2
        self.frame_count = 0
        
        # FPS hesaplaması için
        self.prev_time = time.time()
    
    def arac_turu_guncelle(self, cls, serit_istatistik):
        """
        Araç türüne göre istatistik günceller
        
        Args:
            cls: Araç sınıfı (2=araba, 3=motorsiklet, 5=otobüs, 7=kamyon)
            serit_istatistik: Güncellenecek serit istatistiği
        """
        if cls in self.ARAC_SINIFI:
            arac_turu = self.ARAC_SINIFI[cls]
            serit_istatistik[arac_turu] += 1
    
    def serit_cizgileri_ciz(self, frame):
        """Video frame'e poligon (bölge) çizgilerini çizer"""
        # Sol bölgeyi çiz
        cv2.polylines(frame, [self.sol_bolge], isClosed=True, color=(255, 0, 0), thickness=3)
        
        # Sağ bölgeyi çiz
        cv2.polylines(frame, [self.sag_bolge], isClosed=True, color=(255, 0, 0), thickness=3)
    
    def istatistik_kutusu_ciz(self, frame):
        """Video frame'e istatistik kutusunu çizer"""
        cv2.rectangle(frame, (10, 5), (300, 230), (0, 0, 0), -1)
        
        # Sol şerit istatistikleri
        cv2.putText(frame, f"Sol Araclar: {self.sol_sayac}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Araba: {self.sol_serit_istatistik['araba']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Motorsiklet: {self.sol_serit_istatistik['motorsiklet']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Otobus: {self.sol_serit_istatistik['otobüs']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Kamyon: {self.sol_serit_istatistik['kamyon']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Sağ şerit istatistikleri
        cv2.putText(frame, f"Sag Araclar: {self.sag_sayac}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Araba: {self.sag_serit_istatistik['araba']}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Motorsiklet: {self.sag_serit_istatistik['motorsiklet']}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Otobus: {self.sag_serit_istatistik['otobüs']}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Kamyon: {self.sag_serit_istatistik['kamyon']}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def araclar_ciz(self, frame, boxes, track_ids, classes):
        """Tespit edilen araçları frame'e çizer"""
        for box, track_id, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Araç merkezini çember ile göster
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Bounding box çiz
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Araç ID'sini yaz
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def nokta_poligon_icinde_mi(self, nokta, poligon):
        """
        Bir noktanın poligon bölgesi içinde olup olmadığını kontrol eder
        
        Args:
            nokta: (x, y) koordinatı
            poligon: np.array formatında poligon noktaları
            
        Returns:
            True: Nokta poligon içinde
            False: Nokta poligon dışında
        """
        # cv2.pointPolygonTest kullanarak kontrol et
        # Sonuç: >0 içinde, <0 dışında, =0 sınırda
        result = cv2.pointPolygonTest(poligon, nokta, False)
        return result >= 0
    
    def sayac_kontrol(self, boxes, track_ids, classes):
        """Araçları sayacak kısım - Poligon mantığı kullanır"""
        if boxes is None:
            return
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            nokta = (cx, cy)
            
            # Sol bölge kontrol
            if self.nokta_poligon_icinde_mi(nokta, self.sol_bolge):
                if track_id not in self.tracked_ids["sol"]:
                    self.tracked_ids["sol"].add(track_id)
                    self.sol_sayac += 1
                    self.arac_turu_guncelle(cls, self.sol_serit_istatistik)
            
            # Sağ bölge kontrol
            if self.nokta_poligon_icinde_mi(nokta, self.sag_bolge):
                if track_id not in self.tracked_ids["sag"]:
                    self.tracked_ids["sag"].add(track_id)
                    self.sag_sayac += 1
                    self.arac_turu_guncelle(cls, self.sag_serit_istatistik)
    
    def fps_guncelle_ve_ciz(self, frame):
        """FPS'i hesapla ve frame'e çiz"""
        curr_time = time.time()
        fps_display = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS:{int(fps_display)}", (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def run(self):
        """Video oynatma ve araç sayma işlemini başlat"""
        print("Video oynatılıyor... Çıkmak için 'q' tuşuna basın.")
        print(f"Video: {self.genislik}x{self.yukseklik}, {self.fps_video:.2f} FPS")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Frame atlama
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # YOLO ile tespit ve takip
            results = self.model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
            r = results[0]
            
            # Serit çizgilerini çiz
            self.serit_cizgileri_ciz(frame)
            
            # Tespit edilen araçları işle
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                classes = r.boxes.cls.cpu().numpy().astype(int)
                
                # Araçları sayacağına göre kontrol et
                self.sayac_kontrol(boxes, track_ids, classes)
                
                # Araçları frame'e çiz
                self.araclar_ciz(frame, boxes, track_ids, classes)
            
            # İstatistik kutusunu çiz
            self.istatistik_kutusu_ciz(frame)
            
            # FPS'i güncelle ve çiz
            self.fps_guncelle_ve_ciz(frame)
            
            # Frame'i göster
            cv2.imshow("Trafik Sayaci", frame)
            
            # Klavye girişi kontrol et
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        
        # Kaynakları serbest bırak
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Sonuçları yazdır
        self.ozet_yazdir()
    
    def ozet_yazdir(self):
        """Sayım sonuçlarını konsola yazdır"""
        print("\n" + "="*50)
        print("TRAFIK SAYIM SONUÇLARI")
        print("="*50)
        print(f"\nSOL ŞERİT:")
        print(f"  Toplam: {self.sol_sayac}")
        print(f"  Araba: {self.sol_serit_istatistik['araba']}")
        print(f"  Motorsiklet: {self.sol_serit_istatistik['motorsiklet']}")
        print(f"  Otobüs: {self.sol_serit_istatistik['otobüs']}")
        print(f"  Kamyon: {self.sol_serit_istatistik['kamyon']}")
        
        print(f"\nSAĞ ŞERİT:")
        print(f"  Toplam: {self.sag_sayac}")
        print(f"  Araba: {self.sag_serit_istatistik['araba']}")
        print(f"  Motorsiklet: {self.sag_serit_istatistik['motorsiklet']}")
        print(f"  Otobüs: {self.sag_serit_istatistik['otobüs']}")
        print(f"  Kamyon: {self.sag_serit_istatistik['kamyon']}")
        
        toplam = self.sol_sayac + self.sag_sayac
        print(f"\nGENEL TOPLAM: {toplam}")
        print("="*50)


if __name__ == "__main__":
    # Video dosya yolu
    VIDEO_PATH = "C:/Users/EMİRHAN/Downloads/Yolokurs/Data/highwy.mp4"
    
    # Sınıfı başlat ve çalıştır
    sayac = carcount(VIDEO_PATH)
    sayac.run()