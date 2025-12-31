import cv2
import numpy as np
from collections import deque
print("Program Başlıyor...")
WHITE_AUTO_GRAY_THRESH = 220
WHITE_RATIO_THRESHOLD = 0.008

SATURATION_THRESH = 60 

ROI_TOP_RATIO = 0.55
KALMAN_Y_BIN = 10
YELLOW_H_MIN = 15

YELLOW_H_MAX = 40
YELLOW_S_MIN = 90

YELLOW_V_MIN = 90

class KalmanTakip:
    def __init__(self):
        self.x = 0
        self.v = 0
        self.p = 100
        self.q = 1
        self.r = 10
    
    def tahmin(self):
        self.x = self.x + self.v
        self.p = self.p + self.q
        return self.x
    
    def guncelle(self, olcum):
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (olcum - self.x)
        self.v = k * (olcum - self.x)
        self.p = (1 - k) * self.p
        return self.x
    
    def clear(self):
        self.v = 0
        self.p = 100

class YolTakip:
    def __init__(self, video_yolu, isim, yontem="auto"):
        self.cap = cv2.VideoCapture(video_yolu)
        self.isim = isim
        self.yontem = yontem
        self.genislik = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.yukseklik = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.sol_gecmis = deque(maxlen=15)
        self.sag_gecmis = deque(maxlen=15)
        self.son_sol = []
        self.son_sag = []
        
        self.son_yol_genisligi = 0
        self.genislik_gecmis = deque(maxlen=15)
        self.ogrenme_tamamlandi = False
        
        self.roi_y = 0
        self.debug = {}
        self.aktif_yontem = ""

        self.kalman_sol = {}
        self.kalman_sag = {}
        
        self.genislik_m = None
        self.genislik_b = None
        self.referans_genislik_m = None
        self.referans_genislik_b = None
        self.perspektif_ogrenildi = False
        self.ref_min_y = None
        self.ref_max_y = None

    def isle(self, frame):
        h, w = frame.shape[:2]
        self.roi_y = int(h * ROI_TOP_RATIO)
        
        roi = frame.copy()
        roi[:self.roi_y, :] = 0
        self.debug['roi'] = roi.copy()
        
        if self.yontem == "beyaz":
            sol_n, sag_n, maske = self.beyaz_yontem(roi)
        elif self.yontem == "saturation":
            sol_n, sag_n, maske = self.saturation_yontem(roi)
        else:
            sol_n, sag_n, maske = self.otomatik_yontem(roi)

        noktalar_img = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)
        for p in sol_n:
            cv2.circle(noktalar_img, (p[0], p[1]), 3, (0,0,255), -1)
        for p in sag_n:
            cv2.circle(noktalar_img, (p[0], p[1]), 3, (255,0,0), -1)
        self.debug['noktalar'] = noktalar_img
        
        return sol_n, sag_n

    def otomatik_yontem(self, roi):
        print("buradayım - otomatik_yontem()")
        gri = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gri.shape
        
        _, beyaz = cv2.threshold(gri, WHITE_AUTO_GRAY_THRESH, 255, cv2.THRESH_BINARY)
        
        beyaz_oran = np.sum(beyaz > 0) / beyaz.size
        
        alt_bolge = beyaz[int(h*0.7):, :]
        alt_beyaz_oran = np.sum(alt_bolge > 0) / alt_bolge.size
        
        sol_serit = beyaz[:, :int(w*0.4)]
        sag_serit = beyaz[:, int(w*0.6):]
        merkez = beyaz[:, int(w*0.4):int(w*0.6)]
        
        sol_beyaz = np.sum(sol_serit > 0) / sol_serit.size
        sag_beyaz = np.sum(sag_serit > 0) / sag_serit.size
        merkez_beyaz = np.sum(merkez > 0) / merkez.size
        
        serit_var = (sol_beyaz > 0.01 or sag_beyaz > 0.01) and merkez_beyaz < 0.05
        
        if beyaz_oran > WHITE_RATIO_THRESHOLD and alt_beyaz_oran > 0.015 and serit_var:
            self.aktif_yontem = "BEYAZ"
            return self.beyaz_yontem(roi)
        else:
            self.aktif_yontem = "SAT"
            return self.saturation_yontem(roi)

    def beyaz_yontem(self, roi):
    
        self.aktif_yontem = "BEYAZ"
        
        maske = self.beyaz_maske_uret(roi)

        sol_n, sag_n, _, _ = self.beyaz_sinir_bul(maske)
        
        if len(sol_n) < 3 and len(self.son_sol) >= 3:
            sol_n = self.son_sol
        if len(sag_n) < 3 and len(self.son_sag) >= 3:
            sag_n = self.son_sag

        sol_n, sag_n = self.alan_koru_gelismis(sol_n, sag_n, maske)
        
        sol_n = self.kalman_uygula(sol_n, self.kalman_sol)
        sag_n = self.kalman_uygula(sag_n, self.kalman_sag)

        return sol_n, sag_n, maske

    def beyaz_maske_uret(self, roi):
    
        gri = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gri_eq = clahe.apply(gri)
        
        _, white_binary = cv2.threshold(gri_eq, 200, 255, cv2.THRESH_BINARY)
        
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        s_hls = hls[:, :, 2]
        
        white_l = cv2.inRange(l, 180, 255)
        white_s = cv2.inRange(s_hls, 0, 80)
        white_hls = cv2.bitwise_and(white_l, white_s)
        
        white = cv2.bitwise_or(white_binary, white_hls)
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_y = np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN], dtype=np.uint8)
        upper_y = np.array([YELLOW_H_MAX, 255, 255], dtype=np.uint8)
        yellow = cv2.inRange(hsv, lower_y, upper_y)

        maske0 = cv2.bitwise_or(white, yellow)

        k_close = np.ones((5, 3), np.uint8)
        maske = cv2.morphologyEx(maske0, cv2.MORPH_CLOSE, k_close)
        
        k_open = np.ones((3, 3), np.uint8)
        maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, k_open)

        self.debug['hsv'] = cv2.cvtColor(gri_eq, cv2.COLOR_GRAY2BGR)
        self.debug['sat'] = cv2.cvtColor(white_binary, cv2.COLOR_GRAY2BGR)
        self.debug['thresh'] = cv2.cvtColor(maske0, cv2.COLOR_GRAY2BGR)
        self.debug['maske'] = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)

        return maske
    
    
    def alan_koru_gelismis(self, sol_n, sag_n, maske=None):
    
        if len(sol_n) >= 3 and len(sag_n) >= 3:
            sol_arr = np.array(sol_n)
            sag_arr = np.array(sag_n)
            
            min_y = max(sol_arr[:, 1].min(), sag_arr[:, 1].min())
            max_y = min(sol_arr[:, 1].max(), sag_arr[:, 1].max())
            
            if self.ref_min_y is None:
                self.ref_min_y = min_y
                self.ref_max_y = max_y
            
            genislik_y_pairs = []
            for sp in sol_n:
                for gp in sag_n:
                    if abs(sp[1] - gp[1]) < 15:
                        genislik_y_pairs.append((sp[1], abs(gp[0] - sp[0])))
            
            if len(genislik_y_pairs) >= 4:
                y_arr = np.array([p[0] for p in genislik_y_pairs])
                g_arr = np.array([p[1] for p in genislik_y_pairs])
                
                try:
                    coeffs = np.polyfit(y_arr, g_arr, 1)
                    new_m, new_b = coeffs[0], coeffs[1]
                    
                    if not self.perspektif_ogrenildi:
                        self.referans_genislik_m = new_m
                        self.referans_genislik_b = new_b
                        self.genislik_m = new_m
                        self.genislik_b = new_b
                        self.perspektif_ogrenildi = True
                    else:
                        alpha = 0.15
                        self.genislik_m = alpha * new_m + (1 - alpha) * self.genislik_m
                        self.genislik_b = alpha * new_b + (1 - alpha) * self.genislik_b
                except (np.linalg.LinAlgError, ValueError):
                    pass
            
            if self.perspektif_ogrenildi:
                sol_n, sag_n = self._perspektif_sinirla(sol_n, sag_n, maske)
            
            if genislik_y_pairs:
                mevcut_genislik = int(np.median([p[1] for p in genislik_y_pairs]))
                
                if self.ogrenme_tamamlandi and self.son_yol_genisligi > 0:
                    sapma = abs(mevcut_genislik - self.son_yol_genisligi) / self.son_yol_genisligi
                    if sapma > 0.3:
                        return self.son_sol if len(self.son_sol) >= 3 else sol_n, \
                               self.son_sag if len(self.son_sag) >= 3 else sag_n
                
                self.genislik_gecmis.append(mevcut_genislik)
                self.son_yol_genisligi = int(np.mean(self.genislik_gecmis))
                
                if len(self.genislik_gecmis) >= 10:
                    self.ogrenme_tamamlandi = True
            
            return sol_n, sag_n
        
        if not self.perspektif_ogrenildi or self.son_yol_genisligi == 0:
            if len(self.son_sol) >= 3:
                sol_n = self.son_sol
            if len(self.son_sag) >= 3:
                sag_n = self.son_sag
            return sol_n, sag_n

        if len(sag_n) < 3 and len(sol_n) >= 3:
            yeni_sag = []
            for p in sol_n:
                y = p[1]
                beklenen_genislik = self._perspektif_genislik(y)
                yeni_sag.append([int(p[0] + beklenen_genislik), y])
            return sol_n, yeni_sag

        if len(sol_n) < 3 and len(sag_n) >= 3:
            yeni_sol = []
            for p in sag_n:
                y = p[1]
                beklenen_genislik = self._perspektif_genislik(y)
                yeni_sol.append([int(p[0] - beklenen_genislik), y])
            return yeni_sol, sag_n

        if len(self.son_sol) >= 3:
            sol_n = self.son_sol
        if len(self.son_sag) >= 3:
            sag_n = self.son_sag
        
        return sol_n, sag_n
    
    def _perspektif_sinirla(self, sol_n, sag_n, maske=None):
    
        if not self.perspektif_ogrenildi:
            return sol_n, sag_n
        
        sol_skor = self._serit_guvenirligi(sol_n, maske)
        sag_skor = self._serit_guvenirligi(sag_n, maske)
        
        yeni_sol = []
        yeni_sag = []
        
        guven_farki = abs(sag_skor - sol_skor)
        
        if sag_skor > sol_skor and guven_farki > 15:
            for gp in sag_n:
                y = gp[1]
                beklenen_genislik = self._perspektif_genislik(y)
                yeni_sag.append(gp)
                yeni_sol.append([int(gp[0] - beklenen_genislik), y])
        elif sol_skor > sag_skor and guven_farki > 15:
            for sp in sol_n:
                y = sp[1]
                beklenen_genislik = self._perspektif_genislik(y)
                yeni_sol.append(sp)
                yeni_sag.append([int(sp[0] + beklenen_genislik), y])
        else:
            y_to_sol = {p[1]: p[0] for p in sol_n}
            y_to_sag = {p[1]: p[0] for p in sag_n}
            
            tum_y = sorted(set(list(y_to_sol.keys()) + list(y_to_sag.keys())))
            
            for y in tum_y:
                sol_x = y_to_sol.get(y)
                sag_x = y_to_sag.get(y)
                beklenen_genislik = self._perspektif_genislik(y)
                
                if sol_x is not None and sag_x is not None:
                    mevcut_genislik = sag_x - sol_x
                    
                    if y < self.yukseklik * 0.7:
                        tolerans = 0.25
                    else:
                        tolerans = 0.15
                    
                    min_genislik = beklenen_genislik * (1 - tolerans)
                    max_genislik = beklenen_genislik * (1 + tolerans)
                    
                    if mevcut_genislik < min_genislik or mevcut_genislik > max_genislik:
                        if len(self.son_sol) >= 3 and len(self.son_sag) >= 3:
                            onceki_sol_x = self._onceki_x_bul(y, self.son_sol)
                            onceki_sag_x = self._onceki_x_bul(y, self.son_sag)
                            if onceki_sol_x and onceki_sag_x:
                                yeni_sol.append([onceki_sol_x, y])
                                yeni_sag.append([onceki_sag_x, y])
                                continue
                        merkez = (sol_x + sag_x) / 2
                        yeni_sol.append([int(merkez - beklenen_genislik / 2), y])
                        yeni_sag.append([int(merkez + beklenen_genislik / 2), y])
                    else:
                        yeni_sol.append([sol_x, y])
                        yeni_sag.append([sag_x, y])
                elif sol_x is not None:
                    yeni_sol.append([sol_x, y])
                    yeni_sag.append([int(sol_x + beklenen_genislik), y])
                elif sag_x is not None:
                    yeni_sag.append([sag_x, y])
                    yeni_sol.append([int(sag_x - beklenen_genislik), y])
        
        if yeni_sol and yeni_sag:
            return sorted(yeni_sol, key=lambda p: p[1]), sorted(yeni_sag, key=lambda p: p[1])
        return sol_n, sag_n
    
    def _perspektif_genislik(self, y):
    
        if self.referans_genislik_m is not None and self.referans_genislik_b is not None:
            genislik = self.referans_genislik_m * y + self.referans_genislik_b
            
            if self.ref_min_y is not None and self.ref_max_y is not None:
                min_genislik_ref = self.referans_genislik_m * self.ref_min_y + self.referans_genislik_b
                max_genislik_ref = self.referans_genislik_m * self.ref_max_y + self.referans_genislik_b
                
                min_g = max(20, min_genislik_ref * 0.7)
                max_g = max_genislik_ref * 1.4
            else:
                min_g = 30
                max_g = self.son_yol_genisligi * 1.5 if self.son_yol_genisligi > 0 else 500
            
            return int(np.clip(genislik, min_g, max_g))
        
        return self.son_yol_genisligi if self.son_yol_genisligi > 0 else 200
    
    def kalman_uygula(self, noktalar, kalman_dict):
    
        if len(noktalar) < 2:
            return noktalar
        
        sonuc = []
        for p in noktalar:
            y = p[1]
            x = p[0]

            y_key = (int(y) // KALMAN_Y_BIN) * KALMAN_Y_BIN
            

            if y_key not in kalman_dict:
                kalman_dict[y_key] = KalmanTakip()
                kalman_dict[y_key].x = x  # İlk konumu ayarla

            kalman_dict[y_key].tahmin()  # 1. Tahmin yap
            yeni_x = kalman_dict[y_key].guncelle(x)  # 2. Ölçüm ile güncelle
            

            sonuc.append([int(yeni_x), y])
        
        return sonuc

    def saturation_yontem(self, roi):
    
        self.aktif_yontem = "SAT"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        
        hsv_enhanced = cv2.merge([h_ch, cv2.equalizeHist(s_ch), cv2.equalizeHist(v_ch)])
        self.debug['hsv'] = hsv_enhanced.copy()
        
        sat = hsv[:, :, 1]
        self.debug['sat'] = cv2.cvtColor(sat, cv2.COLOR_GRAY2BGR)

        roi_sat = sat[self.roi_y:, :]
        otsu_t, _ = cv2.threshold(roi_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dinamik eşiği sınırlandır (35-160 arası)

        dinamik_t = int(np.clip(max(SATURATION_THRESH, int(otsu_t)), 35, 160))

        _, thresh = cv2.threshold(sat, dinamik_t, 255, cv2.THRESH_BINARY)
        self.debug['thresh'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        maske = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
        self.debug['maske'] = cv2.cvtColor(maske, cv2.COLOR_GRAY2BGR)
        
        sol_n, sag_n = self.sinir_bul(maske)
        
        if len(sol_n) < 3 and len(self.son_sol) >= 3:
            sol_n = self.son_sol
        if len(sag_n) < 3 and len(self.son_sag) >= 3:
            sag_n = self.son_sag

        sol_n, sag_n = self.alan_koru_gelismis(sol_n, sag_n, maske)

        sol_n = self.kalman_uygula(sol_n, self.kalman_sol)
        sag_n = self.kalman_uygula(sag_n, self.kalman_sag)

        return sol_n, sag_n, maske

    def _onceki_x_bul(self, y, onceki_noktalar):
    
        if not onceki_noktalar:
            return None
        en_yakin = min(onceki_noktalar, key=lambda p: abs(p[1] - y))
        if abs(en_yakin[1] - y) < 30:
            return en_yakin[0]
        return None
    
    def _serit_guvenirligi(self, noktalar, maske):
    
        if len(noktalar) < 3:
            return 0.0
        
        skor = 0.0
        
        skor += min(len(noktalar) / 50.0, 1.0) * 30
        
        arr = np.array(noktalar)
        y_span = arr[:, 1].max() - arr[:, 1].min()
        skor += min(y_span / 200.0, 1.0) * 25
        
        try:
            ys = arr[:, 1]
            xs = arr[:, 0]
            if len(noktalar) >= 4:
                coeffs = np.polyfit(ys, xs, min(2, len(noktalar) - 1))
                poly = np.poly1d(coeffs)
                tahminler = poly(ys)
                hata = np.mean(np.abs(xs - tahminler))
                duzgunluk = max(0, 1.0 - hata / 30.0)
                skor += duzgunluk * 25
        except (np.linalg.LinAlgError, ValueError, TypeError):
            pass
        
        if maske is not None and len(noktalar) > 0:
            h, w = maske.shape
            kesintisizlik = 0
            toplam = 0
            for i in range(len(noktalar) - 1):
                y1, y2 = noktalar[i][1], noktalar[i + 1][1]
                if abs(y2 - y1) < 20:
                    x1, x2 = noktalar[i][0], noktalar[i + 1][0]
                    for t in np.linspace(0, 1, 5):
                        y = int(y1 + t * (y2 - y1))
                        x = int(x1 + t * (x2 - x1))
                        if 0 <= y < h and 0 <= x < w:
                            toplam += 1
                            if maske[y, x] > 200:
                                kesintisizlik += 1
            if toplam > 0:
                skor += (kesintisizlik / toplam) * 20
        
        return min(skor, 100.0)
    
    def _merkez_hesapla(self, sol, sag):
    
        if len(sol) < 2 or len(sag) < 2:
            return []
        
        try:
            sol_arr = np.array(sol)
            sag_arr = np.array(sag)
            
            min_y = max(sol_arr[:, 1].min(), sag_arr[:, 1].min())
            max_y = min(sol_arr[:, 1].max(), sag_arr[:, 1].max())
            
            if max_y - min_y < 20:
                return []
            
            sol_poly = None
            sag_poly = None
            
            if len(sol) >= 3:
                try:
                    sol_xs = sol_arr[:, 0]
                    sol_ys = sol_arr[:, 1]
                    if len(sol) >= 6:
                        sol_coeffs = np.polyfit(sol_ys, sol_xs, 2)
                    else:
                        sol_coeffs = np.polyfit(sol_ys, sol_xs, 1)
                    sol_poly = np.poly1d(sol_coeffs)
                except (np.linalg.LinAlgError, ValueError):
                    pass
            
            if len(sag) >= 3:
                try:
                    sag_xs = sag_arr[:, 0]
                    sag_ys = sag_arr[:, 1]
                    if len(sag) >= 6:
                        sag_coeffs = np.polyfit(sag_ys, sag_xs, 2)
                    else:
                        sag_coeffs = np.polyfit(sag_ys, sag_xs, 1)
                    sag_poly = np.poly1d(sag_coeffs)
                except (np.linalg.LinAlgError, ValueError):
                    pass
            
            if sol_poly is None or sag_poly is None:
                orta_noktalar = []
                sol_sorted = sorted(sol, key=lambda p: p[1])
                sag_sorted = sorted(sag, key=lambda p: p[1])
                
                for sp in sol_sorted:
                    for gp in sag_sorted:
                        if abs(sp[1] - gp[1]) < 15:
                            orta_x = (sp[0] + gp[0]) // 2
                            orta_noktalar.append([orta_x, sp[1]])
                            break
                return orta_noktalar
            
            orta_noktalar = []
            for y in range(int(min_y), int(max_y), 5):
                sol_x = int(sol_poly(y))
                sag_x = int(sag_poly(y))
                orta_x = (sol_x + sag_x) // 2
                orta_noktalar.append([orta_x, y])
            
            return orta_noktalar
            
        except Exception as e:
            orta_noktalar = []
            sol_sorted = sorted(sol, key=lambda p: p[1])
            sag_sorted = sorted(sag, key=lambda p: p[1])
            
            for sp in sol_sorted:
                for gp in sag_sorted:
                    if abs(sp[1] - gp[1]) < 15:
                        orta_x = (sp[0] + gp[0]) // 2
                        orta_noktalar.append([orta_x, sp[1]])
                        break
            return orta_noktalar

    def beyaz_sinir_bul(self, maske):
    
        h, w = maske.shape
        lines = cv2.HoughLinesP(maske, 1, np.pi/180, 20, minLineLength=30, maxLineGap=150)
        
        left_points = []
        right_points = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 == y2: continue
                
                m = (x2 - x1) / (y2 - y1)
                if abs(m) > 2.0: continue
                
                if m < 0 and x1 < w * 0.6:
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))
                elif m > 0 and x1 > w * 0.4:
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))

        sol_noktalar = []
        sag_noktalar = []
        
        left_params = None
        right_params = None
        
        if len(left_points) >= 4:
            try:
                xs = np.array([p[0] for p in left_points])
                ys = np.array([p[1] for p in left_points])
                
                y_range = ys.max() - ys.min()
                if y_range < 10:
                    pass
                elif len(left_points) >= 6 and y_range > 50:
                    with np.errstate(all='ignore'):
                        coeffs = np.polyfit(ys, xs, 2)
                        if not np.any(np.isnan(coeffs)) and not np.any(np.isinf(coeffs)):
                            poly = np.poly1d(coeffs)
                            left_params = coeffs
                            
                            for y in range(self.roi_y, h, 2):
                                x = int(poly(y))
                                if 0 <= x < w:
                                    sol_noktalar.append([x, y])
                else:
                    coeffs = np.polyfit(ys, xs, 1)
                    poly = np.poly1d(coeffs)
                    left_params = coeffs
                    
                    for y in range(self.roi_y, h, 2):
                        x = int(poly(y))
                        if 0 <= x < w:
                            sol_noktalar.append([x, y])
            except (np.linalg.LinAlgError, ValueError, TypeError):

                pass

        if len(right_points) >= 4:
            try:
                xs = np.array([p[0] for p in right_points])
                ys = np.array([p[1] for p in right_points])
                
                y_range = ys.max() - ys.min()
                if y_range < 10:
                    pass
                elif len(right_points) >= 6 and y_range > 50:
                    with np.errstate(all='ignore'):
                        coeffs = np.polyfit(ys, xs, 2)
                        if not np.any(np.isnan(coeffs)) and not np.any(np.isinf(coeffs)):
                            poly = np.poly1d(coeffs)
                            right_params = coeffs
                            
                            for y in range(self.roi_y, h, 2):
                                x = int(poly(y))
                                if 0 <= x < w:
                                    sag_noktalar.append([x, y])
                else:
                    coeffs = np.polyfit(ys, xs, 1)
                    poly = np.poly1d(coeffs)
                    right_params = coeffs
                    
                    for y in range(self.roi_y, h, 2):
                        x = int(poly(y))
                        if 0 <= x < w:
                            sag_noktalar.append([x, y])
            except (np.linalg.LinAlgError, ValueError, TypeError):

                pass
                
        return sol_noktalar, sag_noktalar, left_params, right_params

    def sinir_bul(self, maske):
    
        h, w = maske.shape
        merkez = w // 2
        sol_raw, sag_raw = [], []

        for y in range(self.roi_y + 10, h - 30, 5):
            satir = maske[y, :]
            sat_mean = float(np.mean(satir))
            sat_std = float(np.std(satir))
            komsu_esik = float(np.clip(sat_mean - 0.5 * sat_std, 40, 120))
            
            sol_x = None
            for x in range(merkez, 30, -1):
                if satir[x] > 200 and np.mean(satir[x+1:min(x+25, w)]) < komsu_esik:
                    sol_x = x
                    break
            
            sag_x = None
            for x in range(merkez, w - 30):
                if satir[x] > 200 and np.mean(satir[max(0, x-25):x]) < komsu_esik:
                    sag_x = x
                    break
            
            if sol_x:
                sol_raw.append([sol_x, y])
            if sag_x:
                sag_raw.append([sag_x, y])
        
        sol = self._polyfit_noktalar(sol_raw, w)
        sag = self._polyfit_noktalar(sag_raw, w)
        
        return self.filtrele(sol), self.filtrele(sag)
    
    def _polyfit_noktalar(self, noktalar, w):
    
        if len(noktalar) < 4:
            return noktalar
        
        try:
            arr = np.array(noktalar)
            xs = arr[:, 0]
            ys = arr[:, 1]
            
            median_x = np.median(xs)
            mad = np.median(np.abs(xs - median_x))
            if mad > 5:
                mask = np.abs(xs - median_x) < 3 * mad
                xs = xs[mask]
                ys = ys[mask]
            
            if len(xs) < 4:
                return noktalar
            
            if len(xs) >= 8:
                coeffs = np.polyfit(ys, xs, 2)
            else:
                coeffs = np.polyfit(ys, xs, 1)
            
            poly = np.poly1d(coeffs)
            
            min_y = int(ys.min())
            max_y = int(ys.max())
            
            sonuc = []
            for y in range(min_y, max_y, 5):
                x = int(poly(y))
                if 0 <= x < w:
                    sonuc.append([x, y])
            return sonuc
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return noktalar

    def filtrele(self, noktalar):
    
        if len(noktalar) < 5:
            return noktalar
        arr = np.array(noktalar)
        med = np.median(arr[:, 0])
        mad = np.median(np.abs(arr[:, 0] - med))
        if mad < 10:
            mad = max(np.std(arr[:, 0]), 15)
        return [p.tolist() for p in arr if abs(p[0] - med) < 2 * mad] or noktalar

    def yumusat(self, noktalar, gecmis, son_konum):
    
        if len(noktalar) < 2:
            if len(son_konum) >= 2:
                return son_konum
            return []
        
        noktalar = sorted(noktalar, key=lambda p: p[1])
        
        if len(son_konum) > 0:
            filtreli = []
            for p in noktalar:
                min_mesafe = 999
                for sp in son_konum:
                    if abs(p[1] - sp[1]) < 20:
                        mesafe = abs(p[0] - sp[0])
                        min_mesafe = min(min_mesafe, mesafe)
                
                if min_mesafe < 40 or min_mesafe == 999:
                    filtreli.append(p)
            
            if len(filtreli) >= 2:
                noktalar = filtreli
            else:
                return son_konum
        
        noktalar = self.cogunluk_hizala(noktalar)
        
        gecmis.append(noktalar)
        
        if len(gecmis) < 2:
            return noktalar
        
        y_to_x = {}
        for idx, fn in enumerate(gecmis):
            w = 0.15 if idx == len(gecmis) - 1 else 0.85 / max(len(gecmis) - 1, 1)
            for p in fn:
                if p[1] not in y_to_x:
                    y_to_x[p[1]] = []
                y_to_x[p[1]].append((p[0], w))
        
        sonuc = []
        for y in sorted(y_to_x.keys()):
            toplam_x = sum(x * w for x, w in y_to_x[y])
            toplam_w = sum(w for x, w in y_to_x[y])
            if toplam_w > 0:
                sonuc.append([int(toplam_x / toplam_w), y])
        
        return sonuc
    
    def cogunluk_hizala(self, noktalar):
    
        if len(noktalar) < 5:
            return noktalar

        x_degerler = [p[0] for p in noktalar]
        x_median = np.median(x_degerler)

        yakin_noktalar = [p for p in noktalar if abs(p[0] - x_median) < 60]
        
        if len(yakin_noktalar) < 3:
            return noktalar
        
        yakin_arr = np.array(yakin_noktalar)
        if len(yakin_arr) >= 3:
            try:
                if len(yakin_arr) >= 6:
                    z = np.polyfit(yakin_arr[:, 1], yakin_arr[:, 0], 2)
                else:
                    z = np.polyfit(yakin_arr[:, 1], yakin_arr[:, 0], 1)
                poly = np.poly1d(z)
                
                sonuc = []
                for p in noktalar:
                    beklenen_x = poly(p[1])

                    if abs(p[0] - beklenen_x) > 50:
                        sonuc.append([int(beklenen_x), p[1]])
                    else:
                        sonuc.append(p)
                
                return sonuc
            except (np.linalg.LinAlgError, ValueError, TypeError):
                return noktalar
        
        return noktalar

    def kus_bakisi(self, frame, sol, sag):
    
        h, w = frame.shape[:2]

        src = np.float32([
            [w * 0.1, h],
            [w * 0.4, h * 0.6],
            [w * 0.6, h * 0.6],
            [w * 0.9, h]
        ])

        dst = np.float32([
            [w * 0.2, h],
            [w * 0.2, 0],
            [w * 0.8, 0],
            [w * 0.8, h]
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        bird = cv2.warpPerspective(frame, M, (w, h))
        
        if len(sol) >= 2:
            sol_pts = np.array(sol, dtype=np.float32).reshape(-1, 1, 2)
            sol_bird = cv2.perspectiveTransform(sol_pts, M)
            cv2.polylines(bird, [sol_bird.astype(np.int32)], False, (0,0,255), 2)
        
        if len(sag) >= 2:
            sag_pts = np.array(sag, dtype=np.float32).reshape(-1, 1, 2)
            sag_bird = cv2.perspectiveTransform(sag_pts, M)
            cv2.polylines(bird, [sag_bird.astype(np.int32)], False, (255,0,0), 2)
        
        cv2.putText(bird, "BIRD EYE", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        return bird

    def ciz(self, frame, sol, sag):
    
        out = frame.copy()
        if len(sol) >= 2:
            cv2.polylines(out, [np.array(sol, np.int32)], False, (0,0,255), 3)
            for p in sol: cv2.circle(out, (p[0], p[1]), 2, (0,255,255), -1)
        if len(sag) >= 2:
            cv2.polylines(out, [np.array(sag, np.int32)], False, (255,0,0), 3)
            for p in sag: cv2.circle(out, (p[0], p[1]), 2, (0,255,255), -1)
        if len(sol) >= 2 and len(sag) >= 2:
            alan = np.vstack([np.array(sol), np.array(sag)[::-1]]).astype(np.int32)
            ov = out.copy()
            cv2.fillPoly(ov, [alan], (0,255,0))
            cv2.addWeighted(ov, 0.25, out, 0.75, 0, out)
            
            orta_noktalar = self._merkez_hesapla(sol, sag)
            
            if len(orta_noktalar) >= 2:
                cv2.polylines(out, [np.array(orta_noktalar, np.int32)], False, (0,0,255), 2)
        
        cv2.rectangle(out, (5,5), (160,62), (0,0,0), -1)
        cv2.putText(out, self.isim, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(out, f"Yontem: {self.aktif_yontem}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.putText(out, f"S:{len(sol)} G:{len(sag)}", (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(out, f"Genislik: {self.son_yol_genisligi}px", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
        return out

    def sifirla(self):
    
        self.sol_gecmis.clear()
        self.sag_gecmis.clear()
        self.son_sol = []
        self.son_sag = []
        self.son_yol_genisligi = 0
        self.genislik_gecmis.clear()
        self.ogrenme_tamamlandi = False
        self.kalman_sol.clear()
        self.kalman_sag.clear()
        self.genislik_m = None
        self.genislik_b = None
        self.referans_genislik_m = None
        self.referans_genislik_b = None
        self.perspektif_ogrenildi = False
        self.ref_min_y = None
        self.ref_max_y = None

    
    def frame_al(self):
    
        ret, frame = self.cap.read()
        if not ret:
            self.sifirla()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        if not ret or frame is None:
            raise RuntimeError("Video okunamadi! Dosya yolu kontrol edin.")
        
        return frame

def debug_panel_dikey(v, k, bird_img):
    imgs = []
    labels = ["ROI", "HSV", "SAT", "THRESH", "MASKE", "BIRD"]
    keys = ['roi', 'hsv', 'sat', 'thresh', 'maske']
    
    for i, key in enumerate(keys):
        img = v.debug.get(key, np.zeros((v.yukseklik, v.genislik, 3), np.uint8))
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        img = cv2.resize(img, k)
        cv2.putText(img, labels[i], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
        imgs.append(img)
    
    bird_small = cv2.resize(bird_img, k)
    cv2.putText(bird_small, "BIRD", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    imgs.append(bird_small)

    row1 = np.hstack([imgs[0], imgs[1]])
    row2 = np.hstack([imgs[2], imgs[3]])
    row3 = np.hstack([imgs[4], imgs[5]])
    
    return np.vstack([row1, row2, row3])

def main():
    import os
    yol = os.path.dirname(os.path.abspath(__file__))
    
    video2 = os.path.join(yol, "road (2).mp4")
    video3 = os.path.join(yol, "road (3).mp4")
    
    if not os.path.exists(video2):
        raise FileNotFoundError(f"Video bulunamadi: {video2}")
    if not os.path.exists(video3):
        raise FileNotFoundError(f"Video bulunamadi: {video3}")
    
    v2 = YolTakip(video2, "KOY YOLU 2", "saturation")
    v3 = YolTakip(video3, "OTOBAN", "beyaz")

    print("Yol Takip Sistemi v4.0 - Cikis icin 'q' tusuna basin")
    print("Video 1: Köy Yolu 2 (Saturation Yöntemi)")
    print("Video 2: Otoban (Beyaz/Sarı Şerit Yöntemi)")
    print("="*50)
    
    while True:
        f2, f3 = v2.frame_al(), v3.frame_al()
        
        sol2, sag2 = v2.isle(f2)
        sol3, sag3 = v3.isle(f3)
        
        sol2 = v2.yumusat(sol2, v2.sol_gecmis, v2.son_sol)
        sag2 = v2.yumusat(sag2, v2.sag_gecmis, v2.son_sag)
        v2.son_sol, v2.son_sag = sol2, sag2
        
        sol3 = v3.yumusat(sol3, v3.sol_gecmis, v3.son_sol)
        sag3 = v3.yumusat(sag3, v3.sag_gecmis, v3.son_sag)
        v3.son_sol, v3.son_sag = sol3, sag3
        
        r2 = v2.ciz(f2, sol2, sag2)
        r3 = v3.ciz(f3, sol3, sag3)
        
        b2 = v2.kus_bakisi(f2, sol2, sag2)
        b3 = v3.kus_bakisi(f3, sol3, sag3)

        k_sonuc = (640, 360)
        r2 = cv2.resize(r2, k_sonuc)
        r3 = cv2.resize(r3, k_sonuc)

        k_debug = (160, 120)
        d2 = debug_panel_dikey(v2, k_debug, b2)
        d3 = debug_panel_dikey(v3, k_debug, b3)

        col2 = np.hstack([r2, d2])
        col3 = np.hstack([r3, d3])
        panel = np.vstack([col2, col3])
        
        cv2.imshow("YOL TAKIP - 2 VIDEO", panel)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("\nProgram sonlandırılıyor...")
            break
    
    v2.cap.release()
    v3.cap.release()
    cv2.destroyAllWindows()
    print("Program başarıyla sonlandırıldı.")

if __name__ == "__main__":
    main()
