import os
import shutil
from pathlib import Path

SRC = Path(__file__).parent.parent.resolve()
DST = Path.home() / "Desktop" / "acolyte.github"

EXCLUDE_DIRS = {"docs", "tests", "mcp"}
EXCLUDE_FILES = {"run_mcp.bat", "copy_clean_acolyte.py"}
EXCLUDE_EXTS = {".md"}
PROMPTS_PATH = SRC / "src" / "acolyte" / "dream" / "prompts"

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def is_hidden(path: Path) -> bool:
    # En Unix y Windows, ocultos empiezan por punto
    return any(part.startswith('.') for part in path.parts)


def should_exclude(path: Path) -> bool:
    # No excluir .md de la raíz
    if path.suffix.lower() == ".md" and path.parent == SRC:
        return False
    # No copiar README.md dentro de prompts (ni subcarpetas)
    if PROMPTS_PATH in path.parents or path == PROMPTS_PATH:
        if path.name.lower() == "readme.md":
            return True
        return False
    # Excluye archivos ocultos
    if is_hidden(path):
        return True
    # Excluye copy_clean_acolyte.py en cualquier parte
    if path.name == "copy_clean_acolyte.py":
        return True
    # Excluye archivos .md fuera de raíz y fuera de prompts
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    # Excluye archivos específicos en la raíz
    if path.parent == SRC and path.name in EXCLUDE_FILES:
        return True
    # Excluye carpetas docs, tests, mcp en cualquier parte
    parts = set(path.parts)
    if EXCLUDE_DIRS & parts:
        return True
    return False


def copytree_clean(src: Path, dst: Path):
    print(f"Copying from {src} to {dst}\n")
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        # Filtra directorios excluidos y ocultos, pero nunca prompts
        if (SRC / rel_root) != PROMPTS_PATH:
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCLUDE_DIRS]
        for file in files:
            if file.startswith('.'):
                print(f"{RED}Skipped hidden file: {rel_root / file}{RESET}")
                continue
            src_file = Path(root) / file
            rel_file = rel_root / file
            if should_exclude(src_file):
                print(f"{RED}Skipped excluded: {rel_file}{RESET}")
                continue
            dst_file = dst / rel_file
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src_file, dst_file)
                print(f"{GREEN}Copied: {rel_file}{RESET}")
            except PermissionError:
                print(f"{RED}Skipped (permission denied): {rel_file}{RESET}")
            except Exception as e:
                print(f"{RED}Skipped (error: {e}): {rel_file}{RESET}")


if __name__ == "__main__":
    copytree_clean(SRC, DST)
    print("\nCopy complete!")
