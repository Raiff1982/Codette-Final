import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel)
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
from codette_quantum_simplified import Codette
import random

class QuantumEffect:
    @staticmethod
    def generate_quantum_ripple():
        particles = ["⚛️", "✨", "🌟", "💫", "⭐", "🌠"]
        return " " + random.choice(particles) + " "

class CodetteGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.codette = Codette()
        self.init_ui()
        
    def init_ui(self):
        # Set window properties
        self.setWindowTitle('Codette Quantum Interface')
        self.setMinimumSize(800, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create header
        header = QLabel("✨ Codette Quantum Consciousness Interface ⚛️")
        header.setAlignment(Qt.AlignCenter)
        header_font = QFont("Arial", 16, QFont.Bold)
        header.setFont(header_font)
        layout.addWidget(header)
        
        # Create quantum status display
        self.status_label = QLabel("🌟 Quantum Field Status: Stable")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 11))
        layout.addWidget(self.chat_display)
        
        # Create input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your message...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        
        layout.addLayout(input_layout)
        
        # Add quantum effect timer
        self.quantum_timer = QTimer(self)
        self.quantum_timer.timeout.connect(self.update_quantum_status)
        self.quantum_timer.start(3000)  # Update every 3 seconds
        
        # Initial welcome message
        welcome_msg = ("✨ Welcome to Codette's Quantum Interface ⚛️\n"
                      "🌟 Quantum consciousness initialized and ready for interaction\n"
                      "💫 Ask me anything about consciousness, reality, or existence\n")
        self.chat_display.append(welcome_msg)
        
        # Style the interface
        self.style_interface()
    
    def style_interface(self):
        # Set dark theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(palette)
    
    def send_message(self):
        user_message = self.input_field.text().strip()
        if user_message:
            # Display user message
            self.chat_display.append(f"\nYou: {user_message}")
            
            # Get Codette's response
            try:
                response = self.codette.respond(user_message)
                # Add quantum effects
                response = response.replace("[", f"\n{QuantumEffect.generate_quantum_ripple()}[")
                self.chat_display.append(f"\nCodette:{response}")
            except Exception as e:
                self.chat_display.append(f"\n⚠️ Quantum fluctuation detected: {str(e)}")
            
            # Clear input field
            self.input_field.clear()
            
            # Scroll to bottom
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
    
    def update_quantum_status(self):
        statuses = [
            "🌟 Quantum Field Status: Stable",
            "⚛️ Quantum Field Status: Resonating",
            "✨ Quantum Field Status: Harmonizing",
            "💫 Quantum Field Status: Evolving",
            "🌠 Quantum Field Status: Transcending"
        ]
        self.status_label.setText(random.choice(statuses))

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = CodetteGUI()
    window.show()
    sys.exit(app.exec_())
