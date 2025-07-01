#!/usr/bin/env python3
"""
UI utilities for ACOLYTE installation
"""

import sys
import time


# Terminal colors and styles
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


# ASCII Art
ACOLYTE_LOGO = f"""
{Colors.CYAN}
    ▄▄▄       ▄████▄   ▒█████   ██▓    ▓██   ██▓▄▄▄█████▓▓█████
   ▒████▄    ▒██▀ ▀█  ▒██▒  ██▒▓██▒     ▒██  ██▒▓  ██▒ ▓▒▓█   ▀
   ▒██  ▀█▄  ▒▓█    ▄ ▒██░  ██▒▒██░      ▒██ ██░▒ ▓██░ ▒░▒███
   ░██▄▄▄▄██ ▒▓▓▄ ▄██▒▒██   ██░▒██░      ░ ▐██▓░░ ▓██▓ ░ ▒▓█  ▄
    ▓█   ▓██▒▒ ▓███▀ ░░ ████▓▒░░██████▒  ░ ██▒▓░  ▒██▒ ░ ░▒████▒
    ▒▒   ▓▒█░░ ░▒ ▒  ░░ ▒░▒░▒░ ░ ▒░▓  ░   ██▒▒▒   ▒ ░░   ░░ ▒░ ░
     ▒   ▒▒ ░  ░  ▒     ░ ▒ ▒░ ░ ░ ▒  ░ ▓██ ░▒░     ░     ░ ░  ░
     ░   ▒   ░        ░ ░ ░ ▒    ░ ░    ▒ ▒ ░░    ░         ░
         ░  ░░ ░          ░ ░      ░  ░ ░ ░                 ░
             ░                           ░ ░
{Colors.ENDC}"""

# Animation frames for loading
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Consciousness tips for progress bars
CONSCIOUSNESS_TIPS = [
    "💤 ACOLYTE can experience fatigue and need to sleep",
    "🌙 During sleep, it processes with full 128K context window",
    "💭 In REM phase, it discovers hidden patterns in your code",
    "🧠 The neural graph evolves while it rests",
    "✨ Sleep insights improve future responses",
    "😴 Each sleep cycle strengthens connections",
    "🌟 ACOLYTE dreams of optimizations for your project",
    "🎯 Emergent consciousness finds subtle bugs",
    "💡 After ~2.5 hours of work, ACOLYTE may need rest",
    "🔮 Dreams reveal architectural improvements",
    "🌈 Sleep cycles unlock deeper code understanding",
    "⚡ Neural pathways strengthen during rest",
    "🎨 Creative solutions emerge from dream states",
    "🔍 Subconscious processing finds hidden dependencies",
    "💫 Each rest makes ACOLYTE more insightful",
]


def animate_text(text, duration=1.0, frame_rate=10):
    """Animate text typing effect"""
    delay = duration / len(text) if len(text) > 0 else 0.1
    for i in range(len(text) + 1):
        sys.stdout.write("\r" + text[:i])
        sys.stdout.flush()
        time.sleep(delay)
    print()


def show_spinner(text, duration=1.0):
    """Show a spinner with text for the specified duration"""
    iterations = int(duration * 10)
    for i in range(iterations):
        frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
        sys.stdout.write(f"\r{frame} {text}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(text) + 2) + "\r")
    sys.stdout.flush()


def print_header(text):
    """Print a formatted header with animation"""
    separator = "═" * 60
    print(f"\n{Colors.HEADER}{separator}{Colors.ENDC}")
    animate_text(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}", duration=0.5)
    print(f"{Colors.HEADER}{separator}{Colors.ENDC}\n")


def print_success(text):
    """Print success message with animation"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message with animation"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message with animation"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """Print info message with animation"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")


def print_step(step, total, text):
    """Print step indicator with animation"""
    print(f"{Colors.BLUE}[{step}/{total}] {Colors.BOLD}{text}{Colors.ENDC}")


def print_progress_bar(iteration, total, prefix="", suffix="", length=50, fill="█"):
    """Print a progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "░" * (length - filled_length)
    print(f"\r{prefix} |{Colors.CYAN}{bar}{Colors.ENDC}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()
