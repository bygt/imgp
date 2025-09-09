# color_rerank.py
import numpy as np
from PIL import Image
import cv2
from skimage.color import rgb2lab, deltaE_cie76
import faiss
import os
from typing import List, Dict, Any

def color_rerank(query_image_path: str, top_results: List[Dict], k: int = 50) -> List[Dict]:
    """
    LAB uzayında histogram ve ΔE ile ilk 50 sonucu yeniden sırala
    """
    try:
        # Query görselini LAB'ye çevir
        query_img = Image.open(query_image_path)
        query_lab = rgb2lab(np.array(query_img))
        
        # Histogram hesapla
        query_hist = calculate_lab_histogram(query_lab)
        
        # Her sonuç için renk benzerliği hesapla
        color_scores = []
        for result in top_results[:k]:
            try:
                # Sonuç görselini yükle
                result_img = Image.open(result['image_path'])
                result_lab = rgb2lab(np.array(result_img))
                
                # Histogram hesapla
                result_hist = calculate_lab_histogram(result_lab)
                
                # Histogram benzerliği (Bhattacharyya distance)
                hist_sim = cv2.compareHist(query_hist, result_hist, cv2.HISTCMP_BHATTACHARYYA)
                
                # ΔE hesapla (ortalama renk farkı)
                delta_e = calculate_delta_e(query_lab, result_lab)
                
                # Kombine skor (düşük = daha benzer)
                color_score = hist_sim + (delta_e / 100.0)
                color_scores.append(color_score)
                
            except Exception as e:
                color_scores.append(float('inf'))  # Hata durumunda en düşük skor
        
        # Renk skoruna göre yeniden sırala
        sorted_indices = np.argsort(color_scores)
        reranked_results = [top_results[i] for i in sorted_indices]
        
        return reranked_results[:k]
        
    except Exception as e:
        print(f"❌ Renk re-rank hatası: {e}")
        return top_results[:k]

def calculate_lab_histogram(lab_image: np.ndarray, bins: int = 32) -> np.ndarray:
    """LAB görselinin histogramını hesapla"""
    # L, a, b kanallarını ayır
    L = lab_image[:,:,0]
    a = lab_image[:,:,1] 
    b = lab_image[:,:,2]
    
    # Histogram hesapla
    hist_L = np.histogram(L, bins=bins, range=(0, 100))[0]
    hist_a = np.histogram(a, bins=bins, range=(-128, 127))[0]
    hist_b = np.histogram(b, bins=bins, range=(-128, 127))[0]
    
    # Normalize et
    hist_L = hist_L.astype(np.float32) / np.sum(hist_L)
    hist_a = hist_a.astype(np.float32) / np.sum(hist_a)
    hist_b = hist_b.astype(np.float32) / np.sum(hist_b)
    
    return np.concatenate([hist_L, hist_a, hist_b])

def calculate_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """İki LAB görsel arasındaki ortalama ΔE hesapla"""
    # Her piksel için ΔE hesapla
    delta_e = deltaE_cie76(lab1, lab2)
    return np.mean(delta_e)
    