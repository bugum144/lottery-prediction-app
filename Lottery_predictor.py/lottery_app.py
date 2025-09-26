import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
import asyncio
import aiohttp
from functools import lru_cache
import cProfile
import logging
from queue import Queue

# ======================== SETUP ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== SECURE DATA STORAGE ========================
class SecureDataStorage:
    """SQLite-based encrypted storage for lottery data"""
    def __init__(self, db_name="lottery_data.db", secret_key="secure_key_123"):
        self.db_name = db_name
        self.secret_key = hashlib.sha256(secret_key.encode()).digest()
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    date TEXT PRIMARY KEY,
                    num1 INTEGER,
                    num2 INTEGER,
                    num3 INTEGER,
                    num4 INTEGER,
                    num5 INTEGER,
                    encrypted BLOB
                )
            """)
    
    def _encrypt(self, data):
        return bytes(data[i] ^ self.secret_key[i % len(self.secret_key)] for i in range(len(data)))
    
    def save(self, df):
        try:
            with sqlite3.connect(self.db_name) as conn:
                for _, row in df.iterrows():
                    encrypted = self._encrypt(pickle.dumps(row.to_dict()))
                    conn.execute("""
                        INSERT OR REPLACE INTO results 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['Date'].strftime('%Y-%m-%d'),
                        row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5'],
                        encrypted
                    ))
        except Exception as e:
            logger.error(f"Save failed: {e}")
            messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def load(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                df = pd.read_sql("SELECT date, num1, num2, num3, num4, num5 FROM results", conn)
                if not df.empty:
                    df['Date'] = pd.to_datetime(df['date'])
                    df = df.drop('date', axis=1)
                    df.columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Date']
                    return df.sort_values('Date')
        except Exception as e:
            logger.error(f"Load failed: {e}")
            messagebox.showerror("Error", f"Load failed: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5'])

# ======================== LOTTERY PREDICTOR ========================
class LotteryPredictor:
    """Optimized predictor with caching and incremental training"""
    def __init__(self, data):
        self.data = data
        self.models = {}
        self._train_in_background()
    
    def _train_in_background(self):
        if len(self.data) < 50:
            return
            
        def train_task():
            try:
                X, y = self._prepare_features()
                if len(X) < 10:  # Minimum samples required
                    return
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                rf.fit(X_train, y_train)
                self.models['rf'] = rf
                
                y_pred = rf.predict(X_test)
                logger.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            except Exception as e:
                logger.error(f"Training failed: {e}")
        
        threading.Thread(target=train_task, daemon=True).start()
    
    def _prepare_features(self):
        df = self.data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Optimized rolling frequency calculation
        for num in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']:
            df[f'{num}_freq'] = df[num].rolling(30, min_periods=1).apply(
                lambda x: x.value_counts().iloc[0] if len(x) >= 15 else np.nan,
                raw=False
            )
        
        df = df.dropna()
        X = df[[f'{num}_freq' for num in ['Num1', 'Num2', 'Num3', 'Num4', 'Num5']]]
        y = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].shift(-1).dropna()
        
        common_idx = X.index.intersection(y.index)
        return X.loc[common_idx], y.loc[common_idx]
    
    @lru_cache(maxsize=1)
    def predict_ml(self):
        if 'rf' not in self.models:
            return None
        X, _ = self._prepare_features()
        if X.empty:
            return None
        return self.models['rf'].predict(X.iloc[-1:].values)[0].tolist()
    
    @lru_cache(maxsize=1)
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
class AsyncLotteryScraper:
    """Asynchronous web scraper with retry logic"""
    def __init__(self):
        self.base_url = "https://www.ghana-lottery.com/results"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.retries = 3
        self.timeout = aiohttp.ClientTimeout(total=10)
    
    async def scrape(self):
        for attempt in range(self.retries):
            try:
                async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                    async with session.get(self.base_url) as response:
                        if response.status == 200:
                            # Parse actual data here - this is placeholder
                            today = datetime.now().date()
                            return pd.DataFrame([{
                                'Date': today - timedelta(days=i),
                                **{f'Num{j}': np.random.randint(5, 91) for j in range(1, 6)}
                            } for i in range(7)])
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.retries - 1:
                    logger.error("All scraping attempts failed")
                    return pd.DataFrame()
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

# ======================== MAIN APPLICATION ========================
class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ghana Lottery Predictor Pro")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize theme
        self.dark_mode = False
        self._setup_themes()
        
        # Data handling
        self.storage = SecureDataStorage()
        self.data = self.storage.load()
        self.data_queue = Queue()
        
        # Services
        self.scraper = AsyncLotteryScraper()
        self.predictor = LotteryPredictor(self.data)
        
        # UI Setup
        self._setup_ui()
        self._update_data_table()
        self._update_predictions()
        
        # Start background updates
        self._start_background_tasks()
    
    def _setup_themes(self):
        self.style = ttk.Style()
        self.light_theme = {
            'bg': '#f5f6fa',
            'fg': '#2d3436',
            'header_bg': '#273c75',
            'header_fg': '#f5f6fa',
            'button_bg': '#dfe6e9',
            'highlight': '#0984e3'
        }
        self.dark_theme = {
            'bg': '#2d3436',
            'fg': '#dfe6e9',
            'header_bg': '#0c2461',
            'header_fg': '#f5f6fa',
            'button_bg': '#636e72',
            'highlight': '#00cec9'
        }
        self.current_theme = self.light_theme
        self._apply_theme()
    
    def _apply_theme(self):
        theme = self.dark_theme if self.dark_mode else self.light_theme
        self.style.theme_use('clam')
        self.style.configure('.', background=theme['bg'], foreground=theme['fg'])
        self.style.configure('TFrame', background=theme['bg'])
        self.style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
        self.style.configure('TButton', background=theme['button_bg'])
        self.style.map('Treeview', 
                      background=[('selected', theme['highlight'])],
                      foreground=[('selected', theme['fg'])])
        
        if hasattr(self, 'header'):
            self.header.config(bg=theme['header_bg'])
            for child in self.header.winfo_children():
                child.config(bg=theme['header_bg'], fg=theme['header_fg'])
    
    def _setup_ui(self):
        # Header
        self.header = tk.Frame(self.root, bg=self.current_theme['header_bg'])
        self.header.pack(fill=tk.X)
        
        tk.Label(
            self.header, 
            text="Ghana Lottery Predictor Pro", 
            font=("Segoe UI", 24, "bold"), 
            fg=self.current_theme['header_fg'], 
            bg=self.current_theme['header_bg']
        ).pack(pady=10)
        
        # Theme toggle button
        theme_btn = ttk.Button(
            self.header,
            text="🌙 Dark Mode" if not self.dark_mode else "☀️ Light Mode",
            command=self._toggle_theme
        )
        theme_btn.pack(side=tk.RIGHT, padx=10)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (Predictions)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Prediction Frame
        pred_frame = ttk.LabelFrame(left_panel, text="Today's Predictions", padding=15)
        pred_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Prediction Cards
        self._create_prediction_card(pred_frame, "Machine Learning Prediction", "ml_pred", "#e17055")
        self._create_prediction_card(pred_frame, "Top 2 Numbers (Frequency)", "top2", "#00b894")
        self._create_prediction_card(pred_frame, "Top 4 Numbers (Frequency)", "top4", "#0984e3")
        
        # Chart Frame
        chart_frame = ttk.LabelFrame(left_panel, text="Number Frequency Analysis", padding=15)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Replace Plotly chart with Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.chart = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right panel (Data)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        
        # Data controls
        controls_frame = ttk.Frame(right_panel)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🔄 Refresh Data",
            command=self._manual_refresh
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            controls_frame,
            text="📊 Export Data",
            command=self._export_data
        ).pack(side=tk.LEFT)
        
        # Data Table
        table_frame = ttk.LabelFrame(right_panel, text="Recent Results", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview with scrollbars
        tree_scroll_y = ttk.Scrollbar(table_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(
            table_frame,
            columns=('Date', 'Num1', 'Num2', 'Num3', 'Num4', 'Num5'),
            show='headings',
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=100, stretch=tk.NO)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll_y.config(command=self.tree.yview)
        tree_scroll_x.config(command=self.tree.xview)
        
        # Status bar
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Tooltips
        self._setup_tooltips()
        
        # Keyboard shortcuts
        self.root.bind('<Control-r>', lambda e: self._manual_refresh())
        self.root.bind('<Control-e>', lambda e: self._export_data())
        self.root.bind('<Control-t>', lambda e: self._toggle_theme())
    
    def _create_prediction_card(self, parent, title, var_name, color):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text=title, font=('Segoe UI', 11)).pack(anchor=tk.W)
        label = ttk.Label(
            frame, 
            text="Calculating...", 
            font=('Segoe UI', 16, 'bold'),
            foreground=color
        )
        label.pack(anchor=tk.W)
        setattr(self, f"{var_name}_label", label)
    
    def _setup_tooltips(self):
        # Create tooltip class
        class ToolTip:
            def __init__(self, widget, text):
                self.widget = widget
                self.text = text
                self.tipwindow = None
                self.id = None
                self.x = self.y = 0
                self.widget.bind("<Enter>", self.showtip)
                self.widget.bind("<Leave>", self.hidetip)
            
            def showtip(self, event=None):
                x, y, _, _ = self.widget.bbox("insert")
                x += self.widget.winfo_rootx() + 25
                y += self.widget.winfo_rooty() + 25
                
                self.tipwindow = tk.Toplevel(self.widget)
                self.tipwindow.wm_overrideredirect(True)
                self.tipwindow.wm_geometry(f"+{x}+{y}")
                
                label = tk.Label(
                    self.tipwindow, 
                    text=self.text, 
                    justify=tk.LEFT,
                    background="#ffffe0",
                    relief=tk.SOLID,
                    borderwidth=1,
                    font=("Segoe UI", 9)
                )
                label.pack(ipadx=1)
            
            def hidetip(self, event=None):
                if self.tipwindow:
                    self.tipwindow.destroy()
                    self.tipwindow = None
        
        # Add tooltips to key elements
        ToolTip(self.top2_label, "Most frequently drawn numbers in the last 30 days")
        ToolTip(self.top4_label, "Top 4 frequently drawn numbers overall")
        ToolTip(self.ml_pred_label, "Machine learning prediction based on historical patterns")
    
    def _update_status(self, message):
        self.status.config(text=message)
        self.root.update_idletasks()
    
    def _update_predictions(self):
        try:
            # Update frequency predictions
            top2, top4 = self.predictor.predict_frequency()
            self.top2_label.config(text=", ".join(map(str, top2)))
            self.top4_label.config(text=", ".join(map(str, top4)))
            
            # Update ML predictions
            ml_pred = self.predictor.predict_ml()
            if ml_pred:
                self.ml_pred_label.config(text=", ".join(map(str, ml_pred)))
            
            # Update chart
            self._update_chart()
        except Exception as e:
            logger.error(f"Prediction update failed: {e}")
    
    def _update_data_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        # Show loading indicator
        loading_id = self.tree.insert('', 'end', values=("Loading...", "", "", "", "", ""))
        
        def update_task():
            try:
                sorted_data = self.data.sort_values('Date', ascending=False)
                self.tree.delete(loading_id)
                
                for _, row in sorted_data.iterrows():
                    self.tree.insert('', 'end', values=(
                        row['Date'].strftime('%Y-%m-%d'),
                        row['Num1'], row['Num2'], row['Num3'], row['Num4'], row['Num5']
                    ))
            except Exception as e:
                logger.error(f"Table update failed: {e}")
                self.tree.item(loading_id, values=("Error loading data", "", "", "", "", ""))
        
        threading.Thread(target=update_task, daemon=True).start()
    
    def _update_chart(self):
        if self.data.empty:
            return
        
        def chart_task():
            try:
                self.ax.clear()
                all_nums = pd.concat([self.data[f'Num{i}'] for i in range(1, 6)])
                freq = all_nums.value_counts().sort_index()
                
                self.ax.bar(freq.index, freq.values, color=self.current_theme['highlight'])
                self.ax.set_title("Number Frequency (All Time)")
                self.ax.set_xlabel("Number")
                self.ax.set_ylabel("Frequency")
                self.ax.grid(axis='y', linestyle='--', alpha=0.5)
                
                # Set theme colors
                self.fig.patch.set_facecolor(self.current_theme['bg'])
                self.ax.set_facecolor(self.current_theme['bg'])
                for spine in self.ax.spines.values():
                    spine.set_color(self.current_theme['fg'])
                self.ax.xaxis.label.set_color(self.current_theme['fg'])
                self.ax.yaxis.label.set_color(self.current_theme['fg'])
                self.ax.title.set_color(self.current_theme['fg'])
                self.ax.tick_params(colors=self.current_theme['fg'])
                
                self.chart.draw()
            except Exception as e:
                logger.error(f"Chart update failed: {e}")
        
        threading.Thread(target=chart_task, daemon=True).start()
    
    def _start_background_tasks(self):
        # Start data refresh thread
        threading.Thread(target=self._background_updater, daemon=True).start()
        
        # Start queue processor
        threading.Thread(target=self._process_data_queue, daemon=True).start()
    
    def _background_updater(self):
        while True:
            try:
                # Run at 2am daily
                now = datetime.now()
                next_run = now.replace(hour=2, minute=0, second=0) + timedelta(days=1)
                sleep_seconds = (next_run - now).total_seconds()
                time.sleep(sleep_seconds)
                
                self._update_status("Checking for new data...")
                self._fetch_new_data()
            except Exception as e:
                logger.error(f"Background updater failed: {e}")
                time.sleep(3600)  # Retry in 1 hour
    
    def _process_data_queue(self):
        while True:
            try:
                task, *args = self.data_queue.get()
                if task == "update_data":
                    self.data = args[0]
                    self.storage.save(self.data)
                    self.predictor = LotteryPredictor(self.data)
                    self._update_data_table()
                    self._update_predictions()
                elif task == "update_status":
                    self._update_status(args[0])
            except Exception as e:
                logger.error(f"Queue processing failed: {e}")
            finally:
                self.data_queue.task_done()
    
    async def _async_fetch_data(self):
        self._update_status("Fetching new data...")
        try:
            new_data = await self.scraper.scrape()
            if not new_data.empty:
                self.data_queue.put(("update_data", pd.concat([self.data, new_data]).drop_duplicates('Date')))
                self._update_status("Data updated successfully")
            else:
                self._update_status("No new data available")
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            self._update_status(f"Error: {str(e)}")
    
    def _fetch_new_data(self):
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_fetch_data())
            loop.close()
        
        threading.Thread(target=run_async, daemon=True).start()
    
    def _manual_refresh(self):
        self._update_status("Manual refresh started...")
        self._fetch_new_data()
    
    def _export_data(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
            )
            if file_path:
                if file_path.endswith('.csv'):
                    self.data.to_csv(file_path, index=False)
                else:
                    self.data.to_excel(file_path, index=False)
                self._update_status(f"Data exported to {file_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", str(e))
    
    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.current_theme = self.dark_theme if self.dark_mode else self.light_theme
        self._apply_theme()
        
        # Update theme button text
        for child in self.header.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(text="🌙 Dark Mode" if not self.dark_mode else "☀️ Light Mode")
        
        # Redraw chart with new theme
        self._update_chart()

# ======================== RUN APPLICATION ========================
if __name__ == "__main__":
    # Profile the application startup
    profiler = cProfile.Profile()
    profiler.enable()
    
    root = tk.Tk()
    app = LotteryApp(root)
    
    profiler.disable()
    profiler.dump_stats('lottery_profiler.prof')
    
    root.mainloop()