import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import asyncio
from typing import Callable, Optional
import ttkthemes

class LoadingScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Codette AI")
        
        # Window settings
        self.geometry("300x150")
        self.overrideredirect(True)  # Remove window decorations
        self.configure(bg='#2c2c2c')
        
        # Center the window
        self.center_window()
        
        # Create loading animation
        self.frame = ttk.Frame(self)
        self.frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Logo/Title
        self.title_label = ttk.Label(
            self.frame, 
            text="Codette AI",
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.frame,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(pady=10)
        
        # Status label
        self.status = ttk.Label(
            self.frame,
            text="Initializing...",
            font=("Helvetica", 10)
        )
        self.status.pack(pady=5)
        
        # Start progress bar animation
        self.progress.start(10)
        
    def center_window(self):
        """Center the window on the screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
    def update_status(self, text: str):
        """Update the loading status text"""
        self.status.config(text=text)
        self.update()

class CodetteUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Apply modern theme
        self.style = ttkthemes.ThemedStyle(self)
        self.style.set_theme("equilux")
        
        # Window settings
        self.title("Codette AI Assistant")
        self.geometry("1000x600")
        self.configure(bg='#2c2c2c')
        
        # Show loading screen
        self.loading_screen = LoadingScreen(self)
        self.withdraw()  # Hide main window while loading
        
        # Initialize UI components
        self._init_ui()
        
        # Start loading sequence
        self.after(100, self._load_components)
    
    def _init_ui(self):
        """Initialize the main UI components"""
        # Create main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create top bar with buttons
        self.toolbar = ttk.Frame(self.main_frame)
        self.toolbar.pack(fill='x', pady=(0, 10))
        
        # Add buttons
        self.new_chat_btn = ttk.Button(
            self.toolbar, 
            text="New Chat",
            command=self._new_chat
        )
        self.new_chat_btn.pack(side='left', padx=5)
        
        self.clear_btn = ttk.Button(
            self.toolbar,
            text="Clear",
            command=self._clear_chat
        )
        self.clear_btn.pack(side='left', padx=5)
        
        # Create chat display area
        self.chat_frame = ttk.Frame(self.main_frame)
        self.chat_frame.pack(expand=True, fill='both')
        
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=("Helvetica", 10),
            bg='#363636',
            fg='white'
        )
        self.chat_display.pack(expand=True, fill='both', pady=(0, 10))
        
        # Create input area
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill='x')
        
        self.input_field = ttk.Entry(
            self.input_frame,
            font=("Helvetica", 10)
        )
        self.input_field.pack(side='left', expand=True, fill='x', padx=(0, 10))
        
        self.send_btn = ttk.Button(
            self.input_frame,
            text="Send",
            command=self._send_message
        )
        self.send_btn.pack(side='right')
        
        # Bind enter key to send message
        self.input_field.bind('<Return>', lambda e: self._send_message())
        
        # Status bar
        self.status_bar = ttk.Label(
            self.main_frame,
            text="Ready",
            font=("Helvetica", 9),
            anchor='w'
        )
        self.status_bar.pack(fill='x', pady=(10, 0))
    
    def _load_components(self):
        """Simulate loading of components"""
        steps = [
            ("Initializing core systems...", 1.0),
            ("Loading neural networks...", 1.0),
            ("Configuring UI components...", 0.5),
            ("Starting services...", 0.5),
            ("Ready!", 0.2)
        ]
        
        def execute_steps():
            for text, delay in steps:
                self.loading_screen.update_status(text)
                time.sleep(delay)
            
            # Show main window and destroy loading screen
            self.deiconify()
            self.loading_screen.destroy()
        
        # Run loading sequence in separate thread
        threading.Thread(target=execute_steps).start()
    
    def _new_chat(self):
        """Start a new chat session"""
        self.chat_display.delete(1.0, tk.END)
        self._update_status("New chat session started")
    
    def _clear_chat(self):
        """Clear the chat display"""
        self.chat_display.delete(1.0, tk.END)
        self._update_status("Chat cleared")
    
    def _send_message(self):
        """Handle sending a message"""
        message = self.input_field.get().strip()
        if message:
            # Add user message
            self.chat_display.insert(tk.END, f"You: {message}\n", "user")
            self.chat_display.see(tk.END)
            
            # Clear input field
            self.input_field.delete(0, tk.END)
            
            # Simulate AI processing
            self._update_status("Processing...")
            self.after(1000, lambda: self._receive_response(message))
    
    def _receive_response(self, user_message: str):
        """Handle receiving a response from Codette"""
        # Placeholder response - replace with actual AI response
        response = "I understand you're saying: " + user_message
        self.chat_display.insert(tk.END, f"Codette: {response}\n\n", "assistant")
        self.chat_display.see(tk.END)
        self._update_status("Ready")
    
    def _update_status(self, text: str):
        """Update the status bar text"""
        self.status_bar.config(text=text)

def main():
    app = CodetteUI()
    app.mainloop()

if __name__ == "__main__":
    main()
