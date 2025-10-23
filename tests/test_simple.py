"""
Simple test script to verify PyInstaller build
"""
import sys
import os

def main():
    print("Test executable running")
    print(f"Python version: {sys.version}")
    print(f"Executable path: {sys.executable}")
    input("Press Enter to exit...")

if __name__ == '__main__':
    main()
