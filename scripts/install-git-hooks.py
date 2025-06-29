#!/usr/bin/env python3
"""
Instala los git hooks de ACOLYTE en el repositorio actual.
Los hooks permiten que ACOLYTE reaccione a cambios de Git.
"""
import os
import shutil
import stat
import sys
from pathlib import Path


def install_git_hooks():
    """Instala los hooks en .git/hooks del proyecto"""

    # Verificar que estamos en un repo git
    git_dir = Path(".git")
    if not git_dir.exists():
        print("❌ Error: No se encontró repositorio Git en el directorio actual")
        return False

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    # Directorio donde están nuestros hooks
    source_dir = Path(__file__).parent / "git-hooks"

    hooks_to_install = ["post-commit", "post-merge", "post-checkout", "post-fetch"]

    installed = []
    backed_up = []

    for hook_name in hooks_to_install:
        source = source_dir / hook_name
        dest = hooks_dir / hook_name

        if not source.exists():
            print(f"⚠️  Hook {hook_name} no encontrado en {source_dir}")
            continue

        # Backup si ya existe
        if dest.exists():
            backup = dest.with_suffix(".backup")
            shutil.copy2(dest, backup)
            backed_up.append(hook_name)

        # Copiar el hook
        shutil.copy2(source, dest)

        # Hacer ejecutable
        st = os.stat(dest)
        os.chmod(dest, st.st_mode | stat.S_IEXEC)

        installed.append(hook_name)

    print("\n✅ Hooks de Git instalados correctamente:")
    for hook in installed:
        print(f"   - {hook}")

    if backed_up:
        print("\n📦 Se crearon backups de hooks existentes:")
        for hook in backed_up:
            print(f"   - {hook}.backup")

    print("\n🔧 Configuración:")
    print("   - Los hooks se activarán automáticamente en operaciones Git")
    print("   - Notificarán a ACOLYTE en http://localhost:8000")
    print("   - El puerto se lee de .acolyte si existe")
    print("\n💡 Triggers configurados:")
    print("   - post-commit: Después de cada commit")
    print("   - post-merge: Después de pull/merge")
    print("   - post-checkout: Al cambiar de branch")
    print("   - post-fetch: Al traer cambios del remoto")

    return True


def uninstall_git_hooks():
    """Desinstala los hooks de ACOLYTE"""
    git_dir = Path(".git")
    if not git_dir.exists():
        print("❌ Error: No se encontró repositorio Git")
        return False

    hooks_dir = git_dir / "hooks"

    hooks_to_remove = ["post-commit", "post-merge", "post-checkout", "post-fetch"]

    removed = []
    restored = []

    for hook_name in hooks_to_remove:
        hook_path = hooks_dir / hook_name
        backup_path = hook_path.with_suffix(".backup")

        if hook_path.exists():
            # Verificar que es nuestro hook
            with open(hook_path, "r") as f:
                content = f.read()
                if "ACOLYTE" in content:
                    os.remove(hook_path)
                    removed.append(hook_name)

                    # Restaurar backup si existe
                    if backup_path.exists():
                        shutil.move(backup_path, hook_path)
                        restored.append(hook_name)

    if removed:
        print("\n✅ Hooks de ACOLYTE desinstalados:")
        for hook in removed:
            print(f"   - {hook}")

    if restored:
        print("\n♻️  Hooks originales restaurados:")
        for hook in restored:
            print(f"   - {hook}")

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_git_hooks()
    else:
        install_git_hooks()
