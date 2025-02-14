import sys
import tty
import termios
import subprocess
import os

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    print("""
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠀⠀⠀⠀⠈⢿⠿⠟⢛⣛⣛⣛⣛⣛⠻⠿⢿⠋⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⡀⢀⣾⣿⣿⣿⠟⠛⠛⠻⣿⣿⣿⣿⡟⠛⠛⠻⢿⣿⣿⣿⡄⢀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣼⣿⣿⡿⠁⠀⠀⠀⢠⠿⠿⠿⠿⡇⠀⠀⠀⠈⢻⣿⣿⣿⠘⢿⠿⢿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⡿⠿⢿⢠⣿⣿⣿⡇⠀⠀⠀⢰⣧⡀⠀⠀⠀⣰⣦⠀⠀⠀⢸⡿⠋⠁⢀⡀⢀⡀⠉⠙⢿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⠋⠁⠀⠀⠀⠀⠉⠻⣿⣿⣄⠀⣀⣼⣟⠿⠗⠰⠿⢿⣿⣄⣀⢀⣾⡇⠀⠆⢈⣀⣀⡀⠲⠀⢸⣿⣿⣿⣿
⠿⠿⠿⠿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠧⠀⠀⠻⠿⠿⠿⠀⠀⠸⠿⠿⠿⠛
⣿⣿⣿⣿⣧⣀⡀⠀⠀⠀⠀⠀⣀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    """)
    print("Welcome to the Panda Hardware Model 🐼\n")
    print("Type 'help' to see available commands.\n")
    while True:
        try:
            print("> ", end='', flush=True)
            command = ""
            while True:
                char = getch()
                if char == '\r' or char == '\n':
                    print()
                    command = command.strip()
                    if command == "test":
                        try:
                            test_script = os.path.join(current_dir, "tests", "test_matmul.py")
                            subprocess.run(["python", test_script], check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error running tests: {e}")
                        except FileNotFoundError:
                            print(f"Error: test_matmul.py not found in {os.path.join(current_dir, 'tests')}")
                    elif command == "help":
                        print("\nCommands:")
                        print("  help  - See Available Commands.")
                        print("  test  - Run the Test Suite.")
                        print("  q     - Quit.\n")
                    break
                elif char.lower() == 'q':
                    print()
                    return
                elif char in ('\x7f', '\b'):
                    if command:
                        command = command[:-1]
                        print('\b \b', end='', flush=True)
                    continue
                print(char, end='', flush=True)
                command += char
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main() 