import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import threading
import time
import sqlite3
import base64
from datetime import datetime
import io
import json
import pytesseract
import subprocess
import platform

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\rithv\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Set TESSDATA_PREFIX to the tessdata directory
os.environ['TESSDATA_PREFIX'] = r"C:\Users\rithv\AppData\Local\Programs\Tesseract-OCR\tessdata"

# Verify the existence of eng.traineddata
tessdata_path = os.path.join(os.environ['TESSDATA_PREFIX'], 'eng.traineddata')
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(
        f"Tesseract OCR failed: {tessdata_path} not found. "
        "Please ensure the 'eng.traineddata' file is in the tessdata directory and download it from "
        "https://github.com/tesseract-ocr/tessdata if missing."
    )

class DatabaseManager:
    def __init__(self, db_name='license_plates.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    plates_count INTEGER,
                    plate_images BLOB,
                    detection_data TEXT
                )
            ''')
            conn.commit()
            conn.close()
            print(f"Database initialized: {self.db_name}")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def save_detection(self, plates_data):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            plate_images_blob = self.serialize_plate_images(plates_data)
            detection_summary = f"Plates detected: {len(plates_data)}"
            for plate in plates_data:
                detection_summary += f"\nPlate {plate['text']}: {plate['w']}x{plate['h']}px at ({plate['x']}, {plate['y']})"
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, plates_count, plate_images, detection_data)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                len(plates_data),
                plate_images_blob,
                detection_summary
            ))
            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()
            print(f"Detection saved to database with ID: {detection_id}")
            return detection_id
        except Exception as e:
            print(f"Database save error: {e}")
            return None
    
    def image_to_blob(self, image_path):
        try:
            with open(image_path, 'rb') as file:
                return file.read()
        except Exception as e:
            print(f"Error converting image to blob: {e}")
            return None
    
    def serialize_plate_images(self, plates_data):
        try:
            plate_images = {}
            for plate in plates_data:
                if os.path.exists(plate['filename']):
                    with open(plate['filename'], 'rb') as file:
                        plate_images[f"plate_{plate['text']}"] = base64.b64encode(file.read()).decode()
                    plate_images[f"filename_{plate['text']}"] = plate['filename']
            return json.dumps(plate_images).encode()
        except Exception as e:
            print(f"Error serializing plate images: {e}")
            return None
    
    def get_recent_detections(self, limit=20):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, plates_count, plate_images, detection_data 
                FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Database query error: {e}")
            return []

    def get_all_detections(self):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, plates_count, plate_images, detection_data 
                FROM detections 
                ORDER BY timestamp DESC
            ''')
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Database query error: {e}")
            return []

    def delete_detection(self, detection_id):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
            conn.commit()
            conn.close()
            print(f"Detection ID {detection_id} deleted from database")
        except Exception as e:
            print(f"Database delete error: {e}")

    def update_plate_text(self, detection_id, old_plate_text, new_plate_text):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            # Fetch the current entry
            cursor.execute('SELECT plate_images, detection_data FROM detections WHERE id = ?', (detection_id,))
            result = cursor.fetchone()
            if not result:
                return False
            plate_images_blob, detection_data = result
            # Update plate_images
            plate_images = json.loads(plate_images_blob.decode())
            if f"plate_{old_plate_text}" in plate_images:
                plate_images[f"plate_{new_plate_text}"] = plate_images.pop(f"plate_{old_plate_text}")
                plate_images[f"filename_{new_plate_text}"] = plate_images.pop(f"filename_{old_plate_text}")
                new_plate_images_blob = json.dumps(plate_images).encode()
                # Update detection_data
                new_detection_data = detection_data.replace(f"Plate {old_plate_text}:", f"Plate {new_plate_text}:")
                # Update the database
                cursor.execute('''
                    UPDATE detections 
                    SET plate_images = ?, detection_data = ? 
                    WHERE id = ?
                ''', (new_plate_images_blob, new_detection_data, detection_id))
                conn.commit()
                conn.close()
                print(f"Updated plate text from {old_plate_text} to {new_plate_text} for detection ID {detection_id}")
                return True
            return False
        except Exception as e:
            print(f"Database update error: {e}")
            return False

class LicensePlateDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition System - xAI")
        self.root.geometry("1280x720")
        self.root.configure(bg='#1E1E2F')  # Dark background for a modern look
        self.root.resizable(True, True)
        
        self.current_image = None
        self.current_image_path = None
        self.detected_plates = []
        self.camera = None
        self.camera_running = False
        self.camera_thread = None
        self.captured_frame = None
        self.db_manager = DatabaseManager()

        # Style configuration for a professional look
        self.style = ttk.Style()
        self.style.configure("TButton", font=('Helvetica', 10, 'bold'), padding=10)
        self.style.configure("Small.TButton", font=('Helvetica', 8, 'bold'), padding=5)
        self.style.configure("TLabel", font=('Helvetica', 12), background='#1E1E2F', foreground='#E0E0E0')
        self.style.configure("TFrame", background='#1E1E2F')
        self.style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'), background='#3A3A5C', foreground='#E0E0E0')
        self.style.configure("Treeview", font=('Helvetica', 10), background='#2D2D44', foreground='#E0E0E0', fieldbackground='#2D2D44')
        self.style.map("TButton", background=[('active', '#4A4A6C')])
        self.style.map("Small.TButton", background=[('active', '#4A4A6C')])

        # Main container
        self.main_frame = tk.Frame(root, bg='#1E1E2F')
        self.main_frame.pack(fill='both', expand=True, padx=15, pady=15)

        self.setup_ui()
        self.start_camera()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.main_frame, bg='#3A3A5C', height=50, relief='raised', bd=2)
        header_frame.pack(fill='x', pady=(0, 10))
        header_label = tk.Label(header_frame, text="License Plate Recognition System", font=('Helvetica', 16, 'bold'), fg='#E0E0E0', bg='#3A3A5C')
        header_label.pack(pady=10)

        # Content container with two panels
        content_container = tk.Frame(self.main_frame, bg='#1E1E2F')
        content_container.pack(fill='both', expand=True)

        # Camera panel (left)
        self.camera_panel = tk.Frame(content_container, bg='#2D2D44', width=640, relief='raised', bd=2)
        self.camera_panel.pack(side='left', fill='both', padx=(0, 10), pady=5)
        self.camera_panel.pack_propagate(False)

        # Right panel (results, manage database button, and history)
        self.right_panel = tk.Frame(content_container, bg='#2D2D44', width=640, relief='raised', bd=2)
        self.right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0), pady=5)
        self.right_panel.pack_propagate(False)

        # Setup individual panels
        self.setup_camera_panel()
        self.setup_right_panel()

        # Footer
        footer_frame = tk.Frame(self.main_frame, bg='#3A3A5C', height=30, relief='raised', bd=2)
        footer_frame.pack(fill='x', side='bottom', pady=(10, 0))
        footer_label = tk.Label(footer_frame, text="Powered by xAI • OpenCV • Tesseract • SQLite", fg='#E0E0E0', bg='#3A3A5C', font=('Helvetica', 8))
        footer_label.pack(expand=True, pady=5)

    def setup_camera_panel(self):
        # Camera title
        camera_title = tk.Label(self.camera_panel, text="📹 Live Camera Feed", font=('Helvetica', 14, 'bold'), fg='#E0E0E0', bg='#2D2D44')
        camera_title.pack(pady=10)

        # Camera feed display
        self.camera_label = tk.Label(self.camera_panel, bg='#1E1E2F', text="Starting camera...", fg='#E0E0E0', font=('Helvetica', 10))
        self.camera_label.pack(pady=5, padx=5, expand=True, fill='both')

        # Status bar
        self.status_frame = tk.Frame(self.camera_panel, bg='#3A3A5C', relief='raised', bd=1)
        self.status_frame.pack(fill='x', pady=(5, 0))
        self.status_label = tk.Label(self.status_frame, text="Camera active - Auto-capturing every 5 seconds", fg='#E0E0E0', bg='#3A3A5C', font=('Helvetica', 10))
        self.status_label.pack(expand=True, pady=5)
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        
        # Control buttons
        control_frame = tk.Frame(self.camera_panel, bg='#2D2D44')
        control_frame.pack(fill='x', pady=(5, 0))
        start_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        start_btn.pack(side='left', padx=5, pady=5)
        stop_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera)
        stop_btn.pack(side='left', padx=5, pady=5)
        capture_btn = ttk.Button(control_frame, text="Manual Capture", command=self.capture_image)
        capture_btn.pack(side='left', padx=5, pady=5)

    def setup_right_panel(self):
        # Split right panel into results, manage database button, and history
        # Results panel (smaller) - 30% of the height
        self.results_panel = tk.Frame(self.right_panel, bg='#2D2D44', height=150)
        self.results_panel.pack(fill='x', padx=3, pady=(0, 3))
        self.results_panel.pack_propagate(False)

        # New frame for the "Manage Database" button
        self.db_button_frame = tk.Frame(self.right_panel, bg='#2D2D44', height=40)
        self.db_button_frame.pack(fill='x', padx=3, pady=(3, 3))
        self.db_button_frame.pack_propagate(False)

        # History panel (larger) - Takes remaining space
        self.history_panel = tk.Frame(self.right_panel, bg='#2D2D44')
        self.history_panel.pack(fill='both', expand=True, padx=3, pady=(3, 0))

        # Results panel (smaller)
        self.results_frame = tk.Frame(self.results_panel, bg='#2D2D44')
        self.results_frame.pack(fill='both', expand=True, pady=5)
        results_title = tk.Label(self.results_frame, text="📊 Detection Results", font=('Helvetica', 12, 'bold'), fg='#E0E0E0', bg='#2D2D44')
        results_title.pack(pady=5)

        self.results_canvas = tk.Canvas(self.results_frame, bg='#2D2D44', highlightthickness=0)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_results = tk.Frame(self.results_canvas, bg='#2D2D44')
        self.scrollable_results.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        self.results_canvas.create_window((0, 0), window=self.scrollable_results, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)
        self.results_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.results_scrollbar.pack(side="right", fill="y", pady=5)
        self.no_results_label = tk.Label(self.scrollable_results, text="No detections yet", fg='#E0E0E0', bg='#2D2D44', font=('Helvetica', 10))
        self.no_results_label.pack(expand=True, pady=10)

        # "Manage Database" button in the new frame
        manage_db_btn = ttk.Button(self.db_button_frame, text="🗄️ Manage Database", command=self.open_database_manager)
        manage_db_btn.pack(pady=5)
        print("Manage Database button created and packed between Results and History.")

        # History panel (larger)
        self.history_frame = tk.Frame(self.history_panel, bg='#2D2D44')
        self.history_frame.pack(fill='both', expand=True, pady=5)

        # History title
        history_title = tk.Label(self.history_frame, text="📚 Detection History", font=('Helvetica', 12, 'bold'), fg='#E0E0E0', bg='#2D2D44')
        history_title.pack(pady=5)

        # Container for Treeview and Scrollbar to ensure proper layout
        tree_container = tk.Frame(self.history_frame, bg='#2D2D44')
        tree_container.pack(fill='both', expand=True, padx=5, pady=(0, 5))

        # Treeview for history
        self.history_tree = ttk.Treeview(tree_container, columns=('Date', 'Time', 'Plate', 'Filename'), show='headings', height=15)
        self.history_tree.heading('Date', text='Date')
        self.history_tree.heading('Time', text='Time')
        self.history_tree.heading('Plate', text='Plate')
        self.history_tree.heading('Filename', text='Filename')
        self.history_tree.column('Date', width=100, anchor='center')
        self.history_tree.column('Time', width=80, anchor='center')
        self.history_tree.column('Plate', width=100, anchor='center')
        self.history_tree.column('Filename', width=200, anchor='w')
        self.history_tree.pack(side='left', fill='both', expand=True)

        # Scrollbar for Treeview
        history_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        history_scrollbar.pack(side='right', fill='y')

        # Update history display
        self.update_history_display()

    def open_database_manager(self):
        # Create a new window for database management
        db_window = Toplevel(self.root)
        db_window.title("Manage Database - License Plate Recognition System")
        db_window.geometry("1000x600")
        db_window.configure(bg='#1E1E2F')
        db_window.resizable(True, True)

        # Style for the new window
        style = ttk.Style(db_window)
        style.configure("TButton", font=('Helvetica', 10, 'bold'), padding=10)
        style.configure("TLabel", font=('Helvetica', 12), background='#1E1E2F', foreground='#E0E0E0')
        style.configure("TFrame", background='#1E1E2F')
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'), background='#3A3A5C', foreground='#E0E0E0')
        style.configure("Treeview", font=('Helvetica', 10), background='#2D2D44', foreground='#E0E0E0', fieldbackground='#2D2D44')
        style.map("TButton", background=[('active', '#4A4A6C')])

        # Main frame for the database window
        main_frame = tk.Frame(db_window, bg='#1E1E2F')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # Header
        header_frame = tk.Frame(main_frame, bg='#3A3A5C', height=50, relief='raised', bd=2)
        header_frame.pack(fill='x', pady=(0, 10))
        header_label = tk.Label(header_frame, text="Database Management", font=('Helvetica', 16, 'bold'), fg='#E0E0E0', bg='#3A3A5C')
        header_label.pack(pady=10)

        # Database table
        table_frame = tk.Frame(main_frame, bg='#2D2D44', relief='raised', bd=2)
        table_frame.pack(fill='both', expand=True, pady=5)

        # Treeview for database entries
        columns = ('ID', 'Date', 'Time', 'Plate', 'Filename', 'Plates Count')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        tree.heading('ID', text='ID')
        tree.heading('Date', text='Date')
        tree.heading('Time', text='Time')
        tree.heading('Plate', text='Plate')
        tree.heading('Filename', text='Filename')
        tree.heading('Plates Count', text='Plates Count')
        tree.column('ID', width=50, anchor='center')
        tree.column('Date', width=100, anchor='center')
        tree.column('Time', width=80, anchor='center')
        tree.column('Plate', width=100, anchor='center')
        tree.column('Filename', width=200, anchor='w')
        tree.column('Plates Count', width=100, anchor='center')
        tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y', pady=5)

        # Load all database entries
        def load_database():
            for item in tree.get_children():
                tree.delete(item)
            all_detections = self.db_manager.get_all_detections()
            for detection in all_detections:
                detection_id, timestamp, plates_count, plate_images_blob, _ = detection
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                if plate_images_blob:
                    plate_images = json.loads(plate_images_blob.decode())
                    for key, value in plate_images.items():
                        if key.startswith('plate_'):
                            plate_text = key.replace('plate_', '')
                            filename_key = f"filename_{plate_text}"
                            filename = plate_images.get(filename_key, 'Unknown')
                            tree.insert('', 'end', values=(
                                detection_id, date_str, time_str, plate_text, filename, plates_count
                            ))
                else:
                    tree.insert('', 'end', values=(
                        detection_id, date_str, time_str, '-', 'No plates', plates_count
                    ))

        load_database()

        # Actions frame
        actions_frame = tk.Frame(main_frame, bg='#2D2D44')
        actions_frame.pack(fill='x', pady=10)

        # Delete selected entry
        def delete_entry():
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("Warning", "Please select an entry to delete.", parent=db_window)
                return
            detection_id = tree.item(selected_item)['values'][0]
            confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete entry ID {detection_id}?", parent=db_window)
            if confirm:
                self.db_manager.delete_detection(detection_id)
                load_database()
                self.update_history_display()

        delete_btn = ttk.Button(actions_frame, text="🗑️ Delete Entry", command=delete_entry)
        delete_btn.pack(side='left', padx=5, pady=5)

        # Edit plate number
        def edit_plate():
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("Warning", "Please select an entry to edit.", parent=db_window)
                return
            values = tree.item(selected_item)['values']
            detection_id = values[0]
            old_plate_text = values[3]
            if old_plate_text == '-':
                messagebox.showwarning("Warning", "This entry has no plate to edit.", parent=db_window)
                return

            # Create a new window for editing
            edit_window = Toplevel(db_window)
            edit_window.title("Edit Plate Number")
            edit_window.geometry("400x200")
            edit_window.configure(bg='#1E1E2F')
            edit_window.resizable(False, False)

            edit_frame = tk.Frame(edit_window, bg='#1E1E2F')
            edit_frame.pack(fill='both', expand=True, padx=15, pady=15)

            label = tk.Label(edit_frame, text=f"Current Plate: {old_plate_text}", font=('Helvetica', 12), fg='#E0E0E0', bg='#1E1E2F')
            label.pack(pady=5)

            new_plate_label = tk.Label(edit_frame, text="New Plate Number:", font=('Helvetica', 10), fg='#E0E0E0', bg='#1E1E2F')
            new_plate_label.pack(pady=5)
            new_plate_entry = ttk.Entry(edit_frame, font=('Helvetica', 10))
            new_plate_entry.pack(pady=5)

            def save_edit():
                new_plate_text = new_plate_entry.get().strip()
                if not new_plate_text:
                    messagebox.showwarning("Warning", "Please enter a new plate number.", parent=edit_window)
                    return
                if self.db_manager.update_plate_text(detection_id, old_plate_text, new_plate_text):
                    messagebox.showinfo("Success", "Plate number updated successfully!", parent=edit_window)
                    load_database()
                    self.update_history_display()
                    edit_window.destroy()
                else:
                    messagebox.showerror("Error", "Failed to update plate number.", parent=edit_window)

            save_btn = ttk.Button(edit_frame, text="Save", command=save_edit)
            save_btn.pack(pady=10)

        edit_btn = ttk.Button(actions_frame, text="✏️ Edit Plate", command=edit_plate)
        edit_btn.pack(side='left', padx=5, pady=5)

        # Refresh button
        refresh_btn = ttk.Button(actions_frame, text="🔄 Refresh", command=load_database)
        refresh_btn.pack(side='left', padx=5, pady=5)

    def start_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera.")
                self.camera_running = False
                return
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            self.root.after(2000, self.schedule_capture)
            self.status_label.configure(text="Camera active - Auto-capturing every 5 seconds")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            self.camera_running = False

    def stop_camera(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_label.configure(image='', text="Camera stopped", fg='#E0E0E0', font=('Helvetica', 10))
        self.status_label.configure(text="Camera stopped")

    def update_camera(self):
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Failed to read frame from camera. Retrying...")
                    time.sleep(0.1)
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_size = (640, 480)
                frame_resized = cv2.resize(frame_rgb, display_size)
                pil_image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(pil_image)
                self.root.after(0, self.update_camera_display, photo, frame)
                time.sleep(0.03)
            except Exception as e:
                print(f"Camera update error: {e}")
                time.sleep(0.1)

    def update_camera_display(self, photo, frame):
        if self.camera_running:
            self.camera_label.configure(image=photo, text='')
            self.camera_label.image = photo
            self.captured_frame = frame

    def schedule_capture(self):
        if self.camera_running:
            if self.captured_frame is None:
                print("No frame available yet. Waiting for the next frame...")
                self.root.after(1000, self.schedule_capture)
                return
            self.capture_image()
            self.root.after(5000, self.schedule_capture)

    def capture_image(self):
        if self.captured_frame is not None:
            self.current_image_path = f'captured_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(self.current_image_path, cv2.cvtColor(self.captured_frame, cv2.COLOR_RGB2BGR))
            self.status_label.configure(text="Image captured - Running detection...")
            self.progress.pack(fill='x', padx=10, pady=5)
            self.progress.start()
            detection_thread = threading.Thread(target=self.run_numberplate_detection)
            detection_thread.daemon = True
            detection_thread.start()
        else:
            print("Capture attempted but no frame available.")
            messagebox.showwarning("Warning", "No frame available to capture. Retrying in 5 seconds...")

    def run_numberplate_detection(self):
        try:
            img = cv2.imread(self.current_image_path)
            if img is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not read image file"))
                self.root.after(0, self.stop_progress)
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade_path = 'haarcascades/haarcascade_russian_plate_number.xml'
            if not os.path.exists(cascade_path):
                cascade_path = 'haarcascade_russian_plate_number.xml'
                if not os.path.exists(cascade_path):
                    self.root.after(0, lambda: messagebox.showerror("Error", "Haarcascade file not found."))
                    self.root.after(0, self.stop_progress)
                    return
            cascade = cv2.CascadeClassifier(cascade_path)
            plates = cascade.detectMultiScale(gray, 1.2, 5)
            print('Number of detected license plates:', len(plates))
            self.detected_plates = []
            for i, (x, y, w, h) in enumerate(plates):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Calculate margins based on image dimensions
                a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
                
                # Extract plate region with margins
                plate = img[y + a:y + h - a, x + b:x + w - b, :]
                
                # Apply morphological operations
                kernel = np.ones((1,1), np.uint8)
                plate = cv2.dilate(plate, kernel, iterations=1)
                plate = cv2.erode(plate, kernel, iterations=1)
                
                # Convert to grayscale
                plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                
                # Apply thresholding to get binary image
                (thresh, plate_binary) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Recognize characters using pytesseract on the binary image
                try:
                    plate_text = pytesseract.image_to_string(plate_binary, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    plate_text = ''.join(c for c in plate_text if c.isalnum()).strip() or f"Unknown_{i+1}"
                except Exception as ocr_error:
                    error_msg = f"Tesseract OCR failed: {str(ocr_error)}."
                    self.root.after(0, lambda: messagebox.showerror("OCR Error", error_msg))
                    plate_text = f"OCR_Failed_{i+1}"
                
                # Save the processed (binary) plate image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_filename = f'plate_{timestamp}_{plate_text}.jpg'
                cv2.imwrite(plate_filename, plate_binary)
                
                # Store plate details
                self.detected_plates.append({
                    'text': plate_text,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'filename': plate_filename,
                    'image': plate_binary
                })
            
            if os.path.exists(self.current_image_path):
                os.remove(self.current_image_path)
            processed_filename = f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(processed_filename, img)
            detection_id = self.db_manager.save_detection(self.detected_plates)
            if os.path.exists(processed_filename):
                os.remove(processed_filename)
            self.root.after(0, self.display_results, detection_id)
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, self.stop_progress)
            if self.current_image_path and os.path.exists(self.current_image_path):
                os.remove(self.current_image_path)

    def stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.configure(text="Camera active - Auto-capturing every 5 seconds")

    def display_results(self, detection_id):
        self.stop_progress()
        for widget in self.scrollable_results.winfo_children():
            widget.destroy()
        num_plates = len(self.detected_plates)
        self.status_label.configure(text=f"✅ Detection complete - {num_plates} plate{'s' if num_plates != 1 else ''} found")
        results_header = tk.Label(self.scrollable_results, 
                                 text=f"🎯 {num_plates} License Plate{'s' if num_plates != 1 else ''} Detected",
                                 fg='#E0E0E0', bg='#2D2D44', font=('Helvetica', 10, 'bold'))
        results_header.pack(pady=5)
        if detection_id:
            db_info = tk.Label(self.scrollable_results, 
                              text=f"💾 Saved to database (ID: {detection_id})",
                              fg='#4ECDC4', bg='#2D2D44', font=('Helvetica', 8))
            db_info.pack(pady=3)
        if self.detected_plates:
            for plate in self.detected_plates:
                plate_frame = tk.Frame(self.scrollable_results, bg='#3A3A5C', relief='raised', bd=1)
                plate_frame.pack(fill='x', pady=2, padx=5)
                info_text = f"Plate {plate['text']}: {plate['w']}×{plate['h']}px at ({plate['x']}, {plate['y']})"
                title_label = tk.Label(plate_frame, text=info_text, 
                                      fg='#E0E0E0', bg='#3A3A5C', font=('Helvetica', 8))
                title_label.pack(pady=2)
                try:
                    if os.path.exists(plate['filename']):
                        plate_img = Image.open(plate['filename'])
                        if plate_img.size[0] > 0:
                            scale_factor = min(100 / plate_img.size[0], 50 / plate_img.size[1])
                            new_size = (max(1, int(plate_img.size[0] * scale_factor)), 
                                       max(1, int(plate_img.size[1] * scale_factor)))
                            plate_img = plate_img.resize(new_size, Image.Resampling.LANCZOS)
                        plate_photo = ImageTk.PhotoImage(plate_img)
                        plate_image_label = tk.Label(plate_frame, image=plate_photo, bg='#3A3A5C')
                        plate_image_label.image = plate_photo
                        plate_image_label.pack(pady=2)
                except Exception as e:
                    error_label = tk.Label(plate_frame, text="Preview error", 
                                          fg='#FF6B6B', bg='#3A3A5C', font=('Helvetica', 6))
                    error_label.pack(pady=2)
        else:
            no_plates_label = tk.Label(self.scrollable_results, text="No license plates detected", 
                                      fg='#FFA726', bg='#2D2D44', font=('Helvetica', 8))
            no_plates_label.pack(pady=10)
        actions_frame = tk.Frame(self.scrollable_results, bg='#2D2D44')
        actions_frame.pack(pady=5)
        capture_again_btn = ttk.Button(actions_frame, text="🔄 Capture Again", 
                                      command=self.reset_for_next_capture, style="Small.TButton")
        capture_again_btn.pack(pady=2)
        view_files_btn = ttk.Button(actions_frame, text="📁 View Files", 
                                   command=self.open_output_folder, style="Small.TButton")
        view_files_btn.pack(pady=2)
        self.update_history_display()

    def update_history_display(self):
        try:
            self.history_tree.delete(*self.history_tree.get_children())
            recent_detections = self.db_manager.get_recent_detections(20)
            for detection in recent_detections:
                detection_id, timestamp, plates_count, plate_images_blob, _ = detection
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                if plate_images_blob:
                    plate_images = json.loads(plate_images_blob.decode())
                    for key, value in plate_images.items():
                        if key.startswith('plate_'):
                            plate_text = key.replace('plate_', '')
                            filename_key = f"filename_{plate_text}"
                            filename = plate_images.get(filename_key, 'Unknown')
                            self.history_tree.insert('', 'end', values=(
                                date_str, time_str, plate_text, filename
                            ))
                else:
                    self.history_tree.insert('', 'end', values=(
                        date_str, time_str, '-', 'No plates'
                    ))
        except Exception as e:
            print(f"Error updating history: {e}")

    def reset_for_next_capture(self):
        self.current_image = None
        self.current_image_path = None
        self.detected_plates = []
        for widget in self.scrollable_results.winfo_children():
            widget.destroy()
        self.no_results_label = tk.Label(self.scrollable_results, text="Ready for next capture", 
                                        fg='#E0E0E0', bg='#2D2D44', font=('Helvetica', 10))
        self.no_results_label.pack(expand=True, pady=10)
        if self.camera_running:
            self.status_label.configure(text="Camera active - Auto-capturing every 5 seconds")
        else:
            self.status_label.configure(text="Camera stopped")

    def open_output_folder(self):
        try:
            current_dir = os.getcwd()
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{current_dir}"')
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", current_dir])
            else:
                subprocess.Popen(["xdg-open", current_dir])
        except Exception as e:
            messagebox.showinfo("Output Location", f"Output files saved in:\n{os.getcwd()}")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = LicensePlateDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    root.mainloop()

if __name__ == "__main__":
    print("Starting License Plate Detector GUI...")
    print("Features:")
    print("- Laptop-friendly UI (16:9, 50/50 split)")
    print("- SQLite database storage (plates only)")
    print("- Timestamp-based plate image naming")
    print("- Columnar history view (Date, Time, Plate Text, Filename)")
    print("- Larger camera feed, always running, auto-capture every 5 seconds")
    print("- Tesseract OCR with enhanced plate processing")
    print("- Manage Database functionality with view, edit, and delete options")
    print("\nRequirements:")
    print("1. haarcascades/haarcascade_russian_plate_number.xml")
    print("2. pip install opencv-python pillow pytesseract")
    print("3. Tesseract OCR installed at C:\\Users\\rithv\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe")
    print("4. tessdata directory with eng.traineddata at C:\\Users\\rithv\\AppData\\Local\\Programs\\Tesseract-OCR\\tessdata")
    print("5. Connected camera")
    main()
