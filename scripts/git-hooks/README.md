# 🪝 Git Hooks de ACOLYTE

Sistema de hooks reactivos que permiten a ACOLYTE responder a cambios en el repositorio del usuario.

## 🎯 Filosofía: Git REACTIVO

ACOLYTE **NO** hace operaciones Git proactivas. En su lugar:
- **Detecta** cuando el usuario hace cambios
- **Reacciona** indexando archivos modificados
- **Notifica** si los cambios afectan el trabajo actual

## 📦 Hooks Implementados

### 1. **post-commit**
Se ejecuta después de cada commit exitoso.

**Detecta**: Archivos modificados en el commit
**Trigger enviado**: `commit`
**Metadata adicional**: mensaje del commit

### 2. **post-merge** 
Se ejecuta después de merge exitoso (incluye `git pull`).

**Detecta**: Todos los archivos cambiados en el merge
**Trigger enviado**: `pull` (siempre, para simplificar)
**Metadata adicional**: 
- `is_pull`: boolean (true si fue pull, false si fue merge local)
- `is_merge_local`: boolean (true si fue merge local)
- `invalidate_cache`: true (importante para limpiar cache)

### 3. **post-checkout**
Se ejecuta al cambiar de branch/tag.

**Detecta**: Archivos diferentes entre el branch anterior y el nuevo
**Trigger enviado**: `checkout`
**Metadata adicional**:
- `current_branch`: nombre del branch actual
- `previous_head`, `new_head`: commits involucrados
- `context_switch`: true

### 4. **post-fetch**
Se ejecuta después de `git fetch` (trae cambios sin aplicar).

**Detecta**: Archivos con cambios disponibles en upstream
**Trigger enviado**: `fetch`
**Metadata adicional**:
- `commits_behind`: número de commits pendientes
- `commits_ahead`: número de commits locales
- `action_required`: true si hay cambios por aplicar

## 🔧 Instalación

```bash
# Instalar hooks en el proyecto del usuario
python scripts/install-git-hooks.py

# Desinstalar (restaura hooks originales si existían)
python scripts/install-git-hooks.py uninstall
```

## 🔌 Integración con ACOLYTE

### Endpoint API

Los hooks envían POST requests a:
```
http://localhost:{port}/api/index/git-changes
```

### Formato del Request

```json
{
    "trigger": "commit|pull|checkout|fetch",
    "files": ["lista", "de", "archivos", "modificados"],
    "metadata": {
        "event": "post-commit|post-merge|post-checkout|post-fetch",
        // ... metadata específica según el hook
    }
}
```

### Flujo de Procesamiento

```
Git Hook detecta cambios
        ↓
HTTP POST con trigger específico  
        ↓
API endpoint recibe request
        ↓
IndexingService orquesta:
  - Chunking de archivos
  - EnrichmentService con trigger
  - Embeddings
  - Guardado en Weaviate
  - 🧠 Actualización del Grafo Neuronal
        ↓
Grafo Neuronal actualiza:
  - Nuevos nodos (archivos/funciones)
  - Relaciones (imports/calls)
  - Fortalece conexiones co-modificadas
        ↓
Si afecta trabajo actual:
  ACOLYTE avisa en el chat
```

### 🧠 Actualización del Grafo Neuronal

Los hooks también actualizan el grafo de relaciones entre código:

1. **Nuevos archivos**: Se añaden como nodos al grafo
2. **Imports/Calls detectados**: Se crean edges entre nodos
3. **Co-modificaciones**: Archivos modificados juntos refuerzan sus conexiones
4. **Patrones detectados**: Bugs recurrentes, refactorizaciones comunes

**Almacenamiento del Grafo**:
- **SQLite** (`code_graph_*` tables): Relaciones estructurales precisas
- **Weaviate** (futuro): Embeddings de patrones para búsqueda semántica

## ⚙️ Configuración

Los hooks leen el puerto de ACOLYTE desde `.acolyte`:

```yaml
ports:
  backend: 8000  # Puerto donde escucha la API
```

Si no encuentra el archivo, usa puerto 8000 por defecto.

## 🛡️ Manejo de Errores

Los hooks están diseñados para **NO interrumpir** operaciones Git:

- Si ACOLYTE no está corriendo → Silencioso (no falla)
- Si hay error de red → Timeout 5s y continúa
- Si hay error inesperado → Log y continúa

## 📝 Notas de Implementación

### Detección Inteligente

- **post-commit**: Usa `git diff-tree` para obtener archivos del commit
- **post-merge**: Compara `ORIG_HEAD` vs `HEAD` 
- **post-checkout**: Recibe prev/new HEAD como argumentos de Git
- **post-fetch**: Compara `HEAD` vs `origin/{branch}`

### Cache e Invalidación

- Solo `pull` trigger invalida cache (cambios externos)
- `fetch` NO invalida (cambios no aplicados aún)
- Otros triggers usan cache normalmente

### Seguridad

- NO ejecutan comandos shell arbitrarios
- Usan subprocess con comandos Git específicos
- Validan paths con pathlib
- Solo se comunican con localhost

## 🔍 Debugging

Para ver qué envían los hooks:

```bash
# Temporalmente añadir al hook:
print(f"ACOLYTE DEBUG: Enviando {len(files)} archivos con trigger '{trigger}'")
```

Ver logs de ACOLYTE:
```
tail -f .acolyte/logs/debug.log
```

## ❌ Lo que NO hacen

- **NO** hacen fetch/pull automático
- **NO** modifican el repositorio
- **NO** se conectan a servicios externos
- **NO** bloquean operaciones Git
- **NO** acceden a información sensible

## ✅ Lo que SÍ hacen

- **SÍ** detectan cambios locales inmediatamente
- **SÍ** notifican a ACOLYTE para indexación reactiva
- **SÍ** preservan el flujo normal de Git
- **SÍ** respetan la privacidad del usuario
- **SÍ** funcionan offline (solo localhost)

---

**NOTA**: Estos hooks implementan la Decisión #11 de ser completamente REACTIVOS a las acciones del usuario, sin automatización proactiva.

# ⚠️ ACLARACIÓN IMPORTANTE: Git Hooks y Plataformas

## Los Git Hooks de ACOLYTE son 100% INDEPENDIENTES de la plataforma

### ¿Por qué funcionan con TODAS las plataformas?

Los hooks se ejecutan en el repositorio **LOCAL** del usuario, NO en el servidor remoto:

```
┌─────────────────┐     ┌──────────────────┐
│  Tu PC Local    │     │ Servidor Remoto  │
├─────────────────┤     ├──────────────────┤
│ .git/           │     │ GitHub           │
│   └── hooks/    │     │ GitLab           │
│       ├── post- │     │ Bitbucket        │
│       │   commit│     │ Gitea            │
│       └── ...   │     │ (cualquiera)     │
│                 │     │                  │
│ ↑ AQUÍ CORREN   │     │ NO importa cuál  │
│   LOS HOOKS     │     │ uses             │
└─────────────────┘     └──────────────────┘
```

### Compatibilidad garantizada con:

- ✅ **GitHub** (github.com, GitHub Enterprise)
- ✅ **GitLab** (gitlab.com, self-hosted)
- ✅ **Bitbucket** (bitbucket.org, Bitbucket Server)
- ✅ **Gitea** (cualquier instancia)
- ✅ **Gogs**
- ✅ **Azure DevOps**
- ✅ **AWS CodeCommit**
- ✅ **Git puro** (sin servidor remoto)
- ✅ **Cualquier otro** que use protocolo Git

### ¿Cómo funcionan?

1. **Usuario hace operación Git**:
   ```bash
   git commit -m "fix: bug"    # Dispara post-commit
   git pull origin main        # Dispara post-merge
   git checkout feature/auth   # Dispara post-checkout
   git fetch                   # Dispara post-fetch
   ```

2. **Hook local se ejecuta**:
   - Detecta archivos cambiados
   - NO se conecta al servidor remoto
   - NO necesita API tokens
   - NO depende de webhooks del servidor

3. **Notifica a ACOLYTE localmente**:
   ```
   HTTP POST → localhost:8000/api/index/git-changes
   ```

### Diferencia con Webhooks del servidor

**Webhooks del servidor** (NO usamos):
- Requieren configuración en GitHub/GitLab/etc
- Necesitan endpoint público o túnel
- Requieren autenticación/tokens
- Dependen de la plataforma específica

**Git hooks locales** (SÍ usamos):
- Se instalan una vez localmente
- Funcionan offline
- No requieren configuración remota
- 100% independientes de plataforma

### Ventajas de nuestra aproximación

1. **Privacidad**: Todo queda en tu máquina
2. **Seguridad**: No hay endpoints públicos
3. **Simplicidad**: Una instalación, funciona siempre
4. **Universalidad**: Cualquier servidor Git
5. **Offline**: Funciona sin internet

### Instalación

```bash
# Funciona igual sin importar si usas GitHub, GitLab, etc.
python scripts/install-git-hooks.py
```

---

**TLDR**: Los hooks son de Git LOCAL, no del servidor. Por eso funcionan con CUALQUIER plataforma.
