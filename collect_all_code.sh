
#!/bin/bash

# JADED Platform - Teljes forráskód gyűjtő script
# Minden fájl teljes kódját bemásolja egy txt fájlba

OUTPUT_FILE="all_source_code_complete.txt"

echo "🚀 JADED Platform - Teljes forráskód gyűjtés kezdése..."
echo "📄 Kimeneti fájl: $OUTPUT_FILE"

# Törlés ha már létezik
rm -f "$OUTPUT_FILE"

# Fejléc hozzáadása
cat << 'EOF' > "$OUTPUT_FILE"
================================================================================
JADED PLATFORM - TELJES FORRÁSKÓD GYŰJTEMÉNY
================================================================================
Generálva: $(date)
Platform: Többnyelvű tudományos számítási platform
Architektúra: Mikroszolgáltatás alapú, több programozási nyelv
================================================================================

EOF

# Függvény fájl hozzáadásához
add_file() {
    local file_path="$1"
    local file_name=$(basename "$file_path")
    
    echo "" >> "$OUTPUT_FILE"
    echo "===============================================" >> "$OUTPUT_FILE"
    echo "File: $file_path" >> "$OUTPUT_FILE"
    echo "===============================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Teljes fájl tartalom hozzáadása
    cat "$file_path" >> "$OUTPUT_FILE"
    
    echo "" >> "$OUTPUT_FILE"
    echo "--- End of $file_path ---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    echo "✓ Hozzáadva: $file_path"
}

# Kizárandó minták (dependencies, packages, git, node_modules, stb.)
EXCLUDE_PATTERNS=(
    "*/node_modules/*"
    "*/venv/*"
    "*/env/*"
    "*/.venv/*"
    "*/__pycache__/*"
    "*/target/*"
    "*/build/*"
    "*/dist/*"
    "*/.git/*"
    "*/deps/*"
    "*/_build/*"
    "*/lib/deps/*"
    "*/cabal-sandbox/*"
    "*/.stack-work/*"
    "*/nimcache/*"
    "*/zig-cache/*"
    "*/zig-out/*"
    "*/elm-stuff/*"
    "*/.lean/*"
    "*/.cargo/*"
    "*/Cargo.lock"
    "*/package-lock.json"
    "*/yarn.lock"
    "*/composer.lock"
    "*/Pipfile.lock"
    "*/poetry.lock"
    "*/uv.lock"
    "*/.pytest_cache/*"
    "*/.mypy_cache/*"
    "*/.coverage"
    "*/coverage/*"
    "*/.tox/*"
    "*/.nox/*"
    "*/.hypothesis/*"
    "*/htmlcov/*"
    "*/.DS_Store"
    "*/Thumbs.db"
    "*/*.pyc"
    "*/*.pyo"
    "*/*.class"
    "*/*.o"
    "*/*.so"
    "*/*.dylib"
    "*/*.dll"
    "*/*.exe"
    "*/*.obj"
    "*/*.pdb"
    "*/*.idb"
    "*/*.lib"
    "*/*.a"
    "*/*.jar"
    "*/*.war"
    "*/*.ear"
    "*/*.zip"
    "*/*.tar.gz"
    "*/*.rar"
    "*/*.7z"
    "*/.*"
)

# Kizárás ellenőrző függvény
should_exclude() {
    local file_path="$1"
    
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$file_path" == $pattern ]]; then
            return 0  # Kizárás
        fi
    done
    
    # További kizárások
    if [[ "$file_path" == *"/.git"* ]] || \
       [[ "$file_path" == *"/node_modules"* ]] || \
       [[ "$file_path" == *"/__pycache__"* ]] || \
       [[ "$file_path" == *"/venv"* ]] || \
       [[ "$file_path" == *"/.venv"* ]] || \
       [[ "$file_path" == *"/build"* ]] || \
       [[ "$file_path" == *"/dist"* ]] || \
       [[ "$file_path" == *"/target"* ]] || \
       [[ "$file_path" == *"/_build"* ]] || \
       [[ "$file_path" == *"/deps"* ]] || \
       [[ "$file_path" == *"/nimcache"* ]] || \
       [[ "$file_path" == *"/zig-cache"* ]] || \
       [[ "$file_path" == *"/zig-out"* ]] || \
       [[ "$file_path" == *"/.stack-work"* ]] || \
       [[ "$file_path" == *"/cabal-sandbox"* ]] || \
       [[ "$file_path" == *"/.lean"* ]] || \
       [[ "$file_path" == *"/.cargo"* ]]; then
        return 0  # Kizárás
    fi
    
    return 1  # Nem kizárás
}

# Forráskód fájlok gyűjtése
echo "📁 Fájlok keresése..."

# Minden fájl feldolgozása rekurzívan
find . -type f \( \
    -name "*.py" -o \
    -name "*.js" -o \
    -name "*.ts" -o \
    -name "*.jl" -o \
    -name "*.hs" -o \
    -name "*.clj" -o \
    -name "*.ex" -o \
    -name "*.exs" -o \
    -name "*.nim" -o \
    -name "*.zig" -o \
    -name "*.dart" -o \
    -name "*.pl" -o \
    -name "*.lean" -o \
    -name "*.agda" -o \
    -name "*.v" -o \
    -name "*.dfy" -o \
    -name "*.fst" -o \
    -name "*.thy" -o \
    -name "*.tla" -o \
    -name "*.c" -o \
    -name "*.cpp" -o \
    -name "*.cc" -o \
    -name "*.cxx" -o \
    -name "*.h" -o \
    -name "*.hpp" -o \
    -name "*.hxx" -o \
    -name "*.rs" -o \
    -name "*.go" -o \
    -name "*.java" -o \
    -name "*.scala" -o \
    -name "*.kt" -o \
    -name "*.swift" -o \
    -name "*.php" -o \
    -name "*.rb" -o \
    -name "*.lua" -o \
    -name "*.R" -o \
    -name "*.r" -o \
    -name "*.m" -o \
    -name "*.ml" -o \
    -name "*.mli" -o \
    -name "*.fs" -o \
    -name "*.fsx" -o \
    -name "*.fsi" -o \
    -name "*.elm" -o \
    -name "*.purs" -o \
    -name "*.reason" -o \
    -name "*.re" -o \
    -name "*.rei" -o \
    -name "*.sol" -o \
    -name "*.vyper" -o \
    -name "*.cairo" -o \
    -name "*.move" -o \
    -name "*.ijs" -o \
    -name "*.apl" -o \
    -name "*.bqn" -o \
    -name "*.chpl" -o \
    -name "*.f90" -o \
    -name "*.f95" -o \
    -name "*.f03" -o \
    -name "*.f08" -o \
    -name "*.for" -o \
    -name "*.ftn" -o \
    -name "*.fut" -o \
    -name "*.wl" -o \
    -name "*.nb" -o \
    -name "*.forth" -o \
    -name "*.4th" -o \
    -name "*.fs" -o \
    -name "*.odin" -o \
    -name "*.pony" -o \
    -name "*.st" -o \
    -name "*.vhdl" -o \
    -name "*.vhd" -o \
    -name "*.sv" -o \
    -name "*.svh" -o \
    -name "*.html" -o \
    -name "*.htm" -o \
    -name "*.css" -o \
    -name "*.scss" -o \
    -name "*.sass" -o \
    -name "*.less" -o \
    -name "*.xml" -o \
    -name "*.json" -o \
    -name "*.yaml" -o \
    -name "*.yml" -o \
    -name "*.toml" -o \
    -name "*.ini" -o \
    -name "*.cfg" -o \
    -name "*.conf" -o \
    -name "*.config" -o \
    -name "*.properties" -o \
    -name "*.env" -o \
    -name "*.sh" -o \
    -name "*.bash" -o \
    -name "*.zsh" -o \
    -name "*.fish" -o \
    -name "*.bat" -o \
    -name "*.cmd" -o \
    -name "*.ps1" -o \
    -name "*.psm1" -o \
    -name "*.Dockerfile" -o \
    -name "Dockerfile*" -o \
    -name "docker-compose*" -o \
    -name "Makefile" -o \
    -name "makefile" -o \
    -name "CMakeLists.txt" -o \
    -name "*.cmake" -o \
    -name "*.mk" -o \
    -name "*.mak" -o \
    -name "*.build" -o \
    -name "*.gradle" -o \
    -name "*.sbt" -o \
    -name "*.cabal" -o \
    -name "*.stack" -o \
    -name "stack.yaml" -o \
    -name "*.nimble" -o \
    -name "*.toml" -o \
    -name "*.md" -o \
    -name "*.rst" -o \
    -name "*.txt" -o \
    -name "README*" -o \
    -name "LICENSE*" -o \
    -name "CHANGELOG*" -o \
    -name "*.replit" -o \
    -name "replit.nix" \
\) | while read -r file; do
    
    if ! should_exclude "$file"; then
        if [[ -f "$file" ]] && [[ -r "$file" ]]; then
            # Ellenőrizzük hogy nem bináris fájl-e
            if file "$file" | grep -q "text\|empty"; then
                add_file "$file"
            else
                echo "⚠️ Bináris fájl kihagyva: $file"
            fi
        fi
    else
        echo "🚫 Kizárva: $file"
    fi
done

# Konfiguráció fájlok hozzáadása
echo "📋 Konfiguráció fájlok hozzáadása..."

CONFIG_FILES=(
    "pyproject.toml"
    "requirements.txt"
    "package.json"
    "Dockerfile"
    "docker-compose.yml"
    ".replit"
    "replit.nix"
    "replit.md"
    "setup.sh"
    "start.sh"
)

for config_file in "${CONFIG_FILES[@]}"; do
    if [[ -f "$config_file" ]]; then
        add_file "$config_file"
    fi
done

# Végső összesítés
echo "" >> "$OUTPUT_FILE"
echo "===============================================" >> "$OUTPUT_FILE"
echo "GYŰJTÉS BEFEJEZÉSE" >> "$OUTPUT_FILE"
echo "===============================================" >> "$OUTPUT_FILE"
echo "Befejezve: $(date)" >> "$OUTPUT_FILE"
echo "Összesen feldolgozott fájlok száma: $(grep -c "File: " "$OUTPUT_FILE")" >> "$OUTPUT_FILE"

# Statisztikák
file_count=$(grep -c "File: " "$OUTPUT_FILE")
file_size=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "✅ Gyűjtés befejezve!"
echo "📊 Statisztikák:"
echo "   - Fájlok száma: $file_count"
echo "   - Kimeneti fájl mérete: $file_size"
echo "   - Kimeneti fájl: $OUTPUT_FILE"
echo ""
echo "🎉 A teljes forráskód gyűjtemény elkészült!"

# Fájl végrehajthatóvá tétele
chmod +x "$OUTPUT_FILE" 2>/dev/null || true
