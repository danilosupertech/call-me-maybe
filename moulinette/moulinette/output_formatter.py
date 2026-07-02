# ABOUTME: Provides formatted terminal output for grading results.
# ABOUTME: Uses ASCII art and optional colors for clear visual separation.

import json
import os
import sys

# Try to use colorama if available, otherwise fall back to no colors
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


# Check if we're in a terminal that supports colors
def _supports_color() -> bool:
    if not COLORS_AVAILABLE:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


class ColoredOutput:
    """Utility class for consistent formatted terminal output during grading."""

    def __init__(self) -> None:
        self.use_color = _supports_color()

    def _color(self, text: str, fore: str = "", style: str = "") -> str:
        """Apply color if available, otherwise return plain text."""
        if not self.use_color:
            return text
        return f"{fore}{style}{text}{Style.RESET_ALL if COLORS_AVAILABLE else ''}"

    def separator(self) -> None:
        """Print a separator line."""
        line = "-" * 60
        if self.use_color:
            print(Fore.CYAN + line)
        else:
            print(line)

    def success(self, msg: str) -> None:
        """Print a success message."""
        if self.use_color:
            print(f"{Fore.GREEN}{Style.BRIGHT}{msg}")
        else:
            print(f"[OK] {msg}")

    def error(self, msg: str) -> None:
        """Print an error message."""
        if self.use_color:
            print(f"{Fore.RED}{Style.BRIGHT}{msg}")
        else:
            print(f"[ERROR] {msg}")

    def warning(self, msg: str) -> None:
        """Print a warning message."""
        if self.use_color:
            print(f"{Fore.YELLOW}{msg}")
        else:
            print(f"[WARN] {msg}")

    def info(self, msg: str) -> None:
        """Print an info message."""
        if self.use_color:
            print(f"{Fore.WHITE}{msg}")
        else:
            print(msg)

    def expected(self, label: str, value: object) -> None:
        """Print an expected value."""
        formatted = _format_terminal_value(value)
        if self.use_color:
            if "\n" in formatted:
                print(f"{Fore.CYAN}  {label}:{Style.RESET_ALL}")
                print(f"{Fore.WHITE}{formatted}{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}  {label}: {Fore.WHITE}{formatted}")
        else:
            if "\n" in formatted:
                print(f"  [EXPECTED] {label}:")
                print(formatted)
            else:
                print(f"  [EXPECTED] {label}: {formatted}")

    def actual(self, label: str, value: object) -> None:
        """Print an actual value."""
        formatted = _format_terminal_value(value)
        if self.use_color:
            if "\n" in formatted:
                print(f"{Fore.MAGENTA}  {label}:{Style.RESET_ALL}")
                print(f"{Fore.WHITE}{formatted}{Style.RESET_ALL}")
            else:
                print(f"{Fore.MAGENTA}  {label}: {Fore.WHITE}{formatted}")
        else:
            if "\n" in formatted:
                print(f"  [GOT]      {label}:")
                print(formatted)
            else:
                print(f"  [GOT]      {label}: {formatted}")

    def prompt(self, prompt_text: str) -> None:
        """Print the prompt being tested."""
        if self.use_color:
            print(f"{Fore.BLUE}Prompt: {Fore.WHITE}{prompt_text}")
        else:
            print(f"Prompt: {prompt_text}")

    def test_header(self, test_num: int, total: int) -> None:
        """Print a simple test case header."""
        if self.use_color:
            print(f"\n{Fore.CYAN}Test {test_num}/{total}{Style.RESET_ALL}")
        else:
            print(f"\nTest {test_num}/{total}")

    def test_result(self, passed: bool, reason: str = "") -> None:
        """Print a test result indicator."""
        if passed:
            suffix = f": {reason}" if reason else ""
            if self.use_color:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}>>> VALID{suffix} <<<"
                    f"{Style.RESET_ALL}"
                )
            else:
                print(f">>> VALID{suffix} <<<")
        else:
            if self.use_color:
                print(f"{Fore.RED}{Style.BRIGHT}>>> INVALID: {reason} <<<{Style.RESET_ALL}")
            else:
                print(f">>> INVALID: {reason} <<<")

    def summary(self, score: int, total: int) -> None:
        """Print final score summary in a plain layout."""
        pct = (score / total) * 100 if total else 0

        # Determine status
        if pct == 100:
            status = "PERFECT"
            color = Fore.GREEN if self.use_color else ""
        elif pct >= 70:
            status = "PASSED"
            color = Fore.YELLOW if self.use_color else ""
        else:
            status = "FAILED"
            color = Fore.RED if self.use_color else ""

        score_text = f"SCORE: {score}/{total} ({pct:.1f}%)"

        print("\nResult:")
        if self.use_color:
            print(f"{color}{Style.BRIGHT}{status}{Style.RESET_ALL}")
            print(score_text)
        else:
            print(status)
            print(score_text)
        print("")


def _format_terminal_value(value: object) -> str:
    """Return a readable terminal representation.

    Dicts and lists are printed as pretty JSON to avoid Python repr formatting.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
