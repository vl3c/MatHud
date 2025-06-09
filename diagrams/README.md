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
