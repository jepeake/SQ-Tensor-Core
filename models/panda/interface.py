import sys
import tty
import termios
import subprocess
import os
import json

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "src", "core", "panda_config.json")
    
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
                    if command.startswith("test"):
                        tokens = command.split()
                        if "--suite" in tokens:
                            test_script = os.path.join(current_dir, "tests", "test_matmul_suite.py")
                        else:
                            test_script = os.path.join(current_dir, "tests", "test_matmul.py")
                        
                        args = ["python", test_script]

                        if "--verbose" in tokens:
                            args.append("--verbose")
                        
                        try:
                            subprocess.run(args, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error running tests: {e}")
                        except FileNotFoundError:
                            print(f"Error: {os.path.basename(test_script)} not found in {os.path.join(current_dir, 'tests')}")
                    
                    elif command.startswith("config"):
                        tokens = command.split()
                        # If '--e' or '--edit' flag is provided, open the file in an editor.
                        if any(token in ("--e", "--edit") for token in tokens):
                            editor = os.environ.get("EDITOR", "nano")
                            print(f"Opening config in editor ({editor}): {config_file}")
                            try:
                                subprocess.run([editor, config_file], check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Error launching editor: {e}")
                        else:
                            if os.path.exists(config_file):
                                try:
                                    with open(config_file, "r") as f:
                                        config_data = json.load(f)
                                    print(json.dumps(config_data, indent=4))
                                except Exception as e:
                                    print(f"Error reading config file: {e}")
                            else:
                                print(f"Config file not found: {config_file}")
                    
                    elif command == "help":
                        print("\nCommands:")
                        print("  help             - See Available Commands.")
                        print("  test             - Run a Single Test without detailed tile output.")
                        print("  test --verbose   - Run a Single Test with detailed tile output.")
                        print("  test --suite     - Run the complete Test Suite.")
                        print("  config           - Display current configuration.")
                        print("  config --edit    - Edit the configuration file.")
                        print("  q                - Quit.\n")
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