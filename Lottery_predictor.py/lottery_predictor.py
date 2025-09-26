import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
from bs4 import BeautifulSoup
import threading
import time
import pickle
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======================== SECURE DATA STORAGE ========================
class SecureDataStorage:
    """Encrypted storage for lottery data"""
    def __init__(self, filename, secret_key):
        self.filename = filename
        self.secret_key = hashlib.sha256(secret_key.encode()).digest()
    
    def _encrypt(self, data):
        return bytes([data[i] ^ self.secret_key[i % len(self.secret_key)] for i in range(len(data))])
    
    def save(self, data):
        try:
            with open(self.filename, 'wb') as f:
                f.write(self._encrypt(pickle.dumps(data)))
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def load(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'rb') as f:
                    return pickle.loads(self._encrypt(f.read()))
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {str(e)}")
        return None

# ======================== LOTTERY PREDICTOR ========================
class LotteryPredictor:
    """Predicts numbers using ML and frequency analysis"""
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.train_models()
    
    def train_models(self):
        if len(self.data) < 50: return
        
        X, y = self._prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        self.models['rf'] = rf
        
        y_pred = rf.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    def _prepare_features(self):
        df = self.data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Rolling frequency features
        for num in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']:
            df[f'{num}_freq'] = df[num].rolling(30).apply(lambda x: x.value_counts().iloc[0] if len(x) >= 30 else np.nan)
        
        df = df.dropna()
        X = df[[f'{num}_freq' for num in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]]
        y = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].shift(-1).dropna()
        
        common_idx = X.index.intersection(y.index)
        return X.loc[common_idx], y.loc[common_idx]
    
    def predict_ml(self):
        if 'rf' not in self.models: return None
        X, _ = self._prepare_features()
        if X.empty: return None
        return self.models['rf'].predict(X.iloc[-1:].values)[0].tolist()
    
    def predict_frequency(self, recent_weight=0.4):
        all_nums = pd.concat([self.data[f'Num{i}'] for i in range(1, 6)])
        freq = all_nums.value_counts()
        
        recent = self.data[self.data['Date'] >= (datetime.now() - timedelta(days=30)).date()]
        if not recent.empty:
            recent_freq = pd.concat([recent[f'Num{i}'] for i in range(1, 6)]).value_counts()
            combined = (freq * (1 - recent_weight) + recent_freq.reindex(freq.index, fill_value=0) * recent_weight)
            combined = combined.sort_values(ascending=False)
            return combined.head(2).index.tolist(), combined.head(4).index.tolist()
        return freq.head(2).index.tolist(), freq.head(4).index.tolist()

# ======================== WEB SCRAPER ========================
class LotteryScraper:
    """Fetches latest results from Ghana Lottery website"""
    def __init__(self):
        self.base_url = "https://www.ghana-lottery.com/results"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def scrape(self):
        try:
            # Simulate scraping (replace with actual implementation)
            time.sleep(1)
            today = datetime.now().date()
            return pd.DataFrame([{
                'Date': today - timedelta(days=i),
                **{f'Num{j}': np.random.randint(5, 91) for j in range(1, 6)}
            } for i in range(7)])
        except Exception as e:
            print(f"Scraping error: {e}")
            return pd.DataFrame()

# ======================== MAIN APPLICATION ========================
class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ghana Lottery Predictor Pro")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f5f6fa")
        
        # Data handling
        self.storage = SecureDataStorage("lottery_data.bin", "secure_key_123")
        self.data = self.storage.load() or pd.DataFrame(columns=['Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5'])
        
        # Services
        self.scraper = LotteryScraper()
        self.predictor = LotteryPredictor(self.data)
        
        # UI Setup
        self._setup_ui()
        self._update_data_table()
        self._update_predictions()
        
        # Start background updates
        threading.Thread(target=self._background_updater, daemon=True).start()
    
    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#f5f6fa", borderwidth=0)
        style.configure("TFrame", background="#f5f6fa")
        style.configure("TLabel", background="#f5f6fa", font=('Segoe UI', 12))
        style.configure("Treeview", font=('Segoe UI', 11), rowheight=28, fieldbackground="#f5f6fa", background="#f5f6fa")
        style.map("Treeview", background=[('selected', '#dff9fb')])
        style.configure("TButton", font=('Segoe UI', 11), padding=6)
        
        # Header
        header = tk.Frame(self.root, bg="#273c75")
        header.pack(fill=tk.X)
        tk.Label(header, text="Ghana Lottery Predictor Pro", font=("Segoe UI", 24, "bold"), fg="#f5f6fa", bg="#273c75").pack(pady=10)
        
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Prediction Tab
        pred_tab = ttk.Frame(self.notebook)
        self.notebook.add(pred_tab, text="Predictions")
        
        # Results Frame
        results_frame = ttk.LabelFrame(pred_tab, text="Today's Predictions", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Prediction Labels
        ttk.Label(results_frame, text="Top 2 Numbers:", font=('Segoe UI', 13, 'bold')).pack(pady=(10,0))
        self.top2_label = ttk.Label(results_frame, text="", font=('Segoe UI', 18, 'bold'), foreground="#e17055")
        self.top2_label.pack(pady=(0,10))
        
        ttk.Label(results_frame, text="Top 4 Numbers:", font=('Segoe UI', 13, 'bold')).pack()
        self.top4_label = ttk.Label(results_frame, text="", font=('Segoe UI', 18, 'bold'), foreground="#00b894")
        self.top4_label.pack(pady=(0,10))
        
        # Add a chart for frequency visualization
        chart_frame = ttk.LabelFrame(pred_tab, text="Number Frequency Chart", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.fig, self.ax = plt.subplots(figsize=(6,2.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Data Table
        table_frame = ttk.LabelFrame(self.root, text="Recent Results", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree = ttk.Treeview(table_frame, columns=('Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5'), show='headings')
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=120)
        self.tree.pack(fill=tk.BOTH, expand=True)
    
    def _update_predictions(self):
        top2, top4 = self.predictor.predict_frequency()
        self.top2_label.config(text=", ".join(map(str, top2)))
        self.top4_label.config(text=", ".join(map(str, top4)))
        self._update_chart()
    
    def _update_data_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for _, row in self.data.sort_values('Date', ascending=False).iterrows():
            self.tree.insert('', 'end', values=(
                row['Date'].strftime('%Y-%m-%d'),
                row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']
            ))
    
    def _update_chart(self):
        self.ax.clear()
        if not self.data.empty:
            all_nums = pd.concat([self.data[f'Num{i}'] for i in range(1, 6)])
            freq = all_nums.value_counts().sort_index()
            self.ax.bar(freq.index, freq.values, color="#0984e3")
            self.ax.set_title("Number Frequency (All Time)", fontsize=14)
            self.ax.set_xlabel("Number")
            self.ax.set_ylabel("Frequency")
            self.ax.grid(axis='y', linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _background_updater(self):
        while True:
            time.sleep(86400)  # 24 hours
            self._fetch_new_data()
    
    def _fetch_new_data(self):
        new_data = self.scraper.scrape()
        if not new_data.empty:
            self.data = pd.concat([self.data, new_data]).drop_duplicates('Date')
            self.storage.save(self.data)
            self.predictor = LotteryPredictor(self.data)
            self._update_data_table()
            self._update_predictions()

# ======================== RUN APPLICATION ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryApp(root)
    root.mainloop()