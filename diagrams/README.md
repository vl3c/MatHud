# MatHud Diagram Generation System

This directory contains the diagram generation system for the MatHud project, organized for better maintainability and separation of concerns.

## 📁 Directory Structure

```
diagrams/
├── scripts/                      # All generation scripts
│   ├── generate_diagrams.py     # Main diagram generator
│   ├── setup_diagram_tools.py   # Dependency installer
│   └── generate_arch.py         # Architecture diagram helper
├── generated_png/               # PNG diagram outputs
├── generated_svg/               # SVG diagram outputs  
├── WORKFLOW_SUMMARY.md          # Quick reference guide
├── diagram_requirements.txt     # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### From Root Directory (Recommended)
```bash
# 1. Setup (one-time)
cd diagrams/scripts
python setup_diagram_tools.py

# 2. Generate diagrams
cd ../..  # back to root
python generate_diagrams_launcher.py
```

### From Scripts Directory
```bash
# 1. Navigate to scripts
cd diagrams/scripts

# 2. Setup (one-time)
python setup_diagram_tools.py

# 3. Generate diagrams
python generate_diagrams.py
```

## ⚙️ System Setup Requirements

### 1. Install Graphviz (System Package)

The diagram generation requires the **Graphviz system package** (provides the `dot` executable):

#### Windows
```powershell
# Using winget (recommended)
winget install graphviz

# Or download from: https://graphviz.org/download/
```

#### macOS
```bash
# Using Homebrew
brew install graphviz
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# RHEL/CentOS/Fedora
sudo yum install graphviz
# or
sudo dnf install graphviz
```

### 2. Add Graphviz to PATH (Windows)

After installing Graphviz on Windows, you may need to add it to your PATH:

#### Option A: PowerShell (Recommended)
```powershell
# Add to user PATH (permanent)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\Graphviz\bin", [EnvironmentVariableTarget]::User)
```

#### Option B: Command Prompt
```cmd
setx PATH "%PATH%;C:\Program Files\Graphviz\bin"
```

#### Option C: GUI Method
1. Right-click "This PC" → Properties
2. Advanced System Settings → Environment Variables
3. Under "User variables", select PATH → Edit
4. Add: `C:\Program Files\Graphviz\bin`

### 3. Verify Installation

Test that everything is working:

```bash
# Check if dot command is available
dot -V

# Should output something like:
# dot - graphviz version 13.0.0 (20250608.1624)
```

### 4. Install Python Dependencies

```bash
# Install Python packages
pip install -r diagrams/diagram_requirements.txt
```

### 🚨 Troubleshooting

**Problem:** `dot is not available` error

**Solutions:**
1. **Restart your terminal/shell** after PATH changes
2. **Verify Graphviz installation:** Check if `C:\Program Files\Graphviz\bin\dot.exe` exists
3. **Manual PATH check:** Run `$env:PATH -split ';' | Select-String "Graphviz"` (PowerShell)
4. **Re-run setup:** `python diagrams/scripts/setup_diagram_tools.py`

## 📊 Generated Diagrams

✅ **Currently Working:**
- **Class Diagrams** (`generated_png/classes_*.png`) - UML class relationships
- **Package Diagrams** (`generated_png/packages_*.png`) - Module dependencies
- **Module-Specific Diagrams** - Individual component views for:
  - AppManager (application coordination)
  - OpenAI_API (AI service integration)
  - WebDriver (browser automation)
  - Workspace (file management)
  - Routes (HTTP endpoints)
- **Dependency Graph** (`generated_png/dependencies.svg`) - Import relationships
- **Call Graph** - Runtime execution flow (manual generation)

## 🛠️ Tools Used

| Tool | Purpose | Status |
|------|---------|--------|
| **pyreverse** (pylint) | UML class diagrams | ✅ Working |
| **graphviz** | Diagram rendering | ✅ Working |
| **pydeps** | Dependency analysis | ✅ Working |
| **pycallgraph2** | Call graph generation | ✅ Working |
| **diagrams** | Architecture visualization | ⚠️ Optional |

## 📈 Benefits for MatHud

This system provides:

1. **Visual Documentation** - Understand Flask backend relationships
2. **Architecture Insights** - See AI integration patterns
3. **Refactoring Support** - Identify coupling and dependencies
4. **Team Communication** - Share visual project understanding
5. **Maintenance** - Track structural evolution over time

## 🎯 Integration Examples

### Code Reviews
Include generated diagrams to explain structural changes:
```markdown
## Changes Overview
![Class Diagram](diagrams/generated_png/classes_MatHud.png)
```

### Documentation
Reference specific module diagrams:
```markdown
## WebDriver Architecture
![WebDriver](diagrams/generated_png/classes_WebDriver.png)
```

### Presentations
Use package diagrams for high-level overviews:
```markdown
## System Architecture
![Packages](diagrams/generated_png/packages_MatHud_packages.png)
```

## 🔧 Customization

Edit `scripts/generate_diagrams.py` to:
- Add new modules to analyze
- Change output formats (PNG, SVG, DOT)
- Adjust diagram styling
- Focus on specific components

## 📚 Documentation

- **WORKFLOW_SUMMARY.md** - Quick reference and best practices
- **diagram_requirements.txt** - Python package dependencies

## 🎉 Success!

Your MatHud project now has visual documentation tools that:
- ✅ Generate comprehensive diagrams on demand
- ✅ Provide multiple visualization formats
- ✅ Support team collaboration
- ✅ Integrate with development workflow
- ✅ Maintain consistency across platforms

---

**Ready to use!** Start with the launcher: `python generate_diagrams_launcher.py` from the root directory!
