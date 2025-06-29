#!/bin/bash
# Script para preparar el proyecto para GitHub público sin READMEs

echo "🔧 Preparando proyecto para GitHub público (sin READMEs)..."

# Hacer backup del .gitignore actual
if [ -f .gitignore ]; then
    cp .gitignore .gitignore.backup
    echo "✅ Backup creado: .gitignore.backup"
fi

# Combinar .gitignore actual con exclusiones de README
cat .gitignore > .gitignore.temp
echo "" >> .gitignore.temp
echo "# === TEMPORAL PARA GITHUB PUBLICO ===" >> .gitignore.temp
cat .gitignore-github >> .gitignore.temp

# Reemplazar .gitignore
mv .gitignore.temp .gitignore

echo "✅ .gitignore actualizado para excluir READMEs"
echo ""
echo "📝 Próximos pasos:"
echo "1. git add ."
echo "2. git commit -m 'feat: prepare for public release without documentation'"
echo "3. git remote add github https://github.com/TU_USUARIO/acolyte.git"
echo "4. git push github main"
echo ""
echo "⚠️  Para restaurar el .gitignore original después:"
echo "    mv .gitignore.backup .gitignore"
