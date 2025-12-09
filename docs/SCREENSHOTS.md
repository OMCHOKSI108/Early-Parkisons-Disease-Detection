# Screenshots Information

This folder contains SVG and PNG versions of the screenshots referenced in the README.md file. These screenshots showcase the Parkinson's Disease Screening Platform user interface and functionality.

## Files

### SVG Versions (Vector - Scalable)
- **1.svg** - Authentication interface (Login/Signup)
- **2.svg** - Drawing analysis interface
- **3.svg** - Voice analysis interface with ensemble results
- **4.svg** - User dashboard with prediction history

### PNG Versions (Raster - Web-ready)
- **1.png** - Authentication interface (Login/Signup)
- **2.png** - Drawing analysis interface
- **3.png** - Voice analysis interface with ensemble results
- **4.png** - User dashboard with prediction history

## Converting SVG to PNG

If you need to convert the SVG files to PNG, you can use one of the following methods:

### Method 1: Using Inkscape (Windows/Mac/Linux)
```bash
inkscape 1.svg --export-filename=1.png --export-dpi=96
```

### Method 2: Using ImageMagick (Windows/Mac/Linux)
```bash
convert 1.svg 1.png
```

### Method 3: Using FFmpeg (Windows/Mac/Linux)
```bash
ffmpeg -i 1.svg -vf scale=1200:600 1.png
```

### Method 4: Using Online Converter
Visit https://cloudconvert.com/ or https://convertio.co/ to convert SVG to PNG online.

### Method 5: Using Python
```python
from PIL import Image
import cairosvg

# Convert SVG to PNG
cairosvg.svg2png(url="1.svg", write_to="1.png", dpi=96)
```

Install required packages:
```bash
pip install pillow cairosvg
```

## Screenshots Description

### Screenshot 1: Authentication Interface (1.png)
**File**: `assets/pngs/1.png`

This screenshot shows the dual-screen authentication interface with:
- **Left Panel**: User login form with email/username and password fields
- **Right Panel**: User registration form with name, email, username, and password
- **Styling**: Modern white theme with blue gradient header and clean form inputs
- **Features**: Real-time validation, error messages, responsive design

### Screenshot 2: Drawing Analysis (2.png)
**File**: `assets/pngs/2.png`

This screenshot displays the drawing analysis workflow:
- **Upload Section**: Drag-and-drop interface for PNG/JPG spiral drawing images
- **Results Section**: Real-time prediction with confidence visualization
- **Model**: HOG + SVM classifier (66% accuracy)
- **Output**: Healthy/Parkinson prediction with percentage confidence and progress bar
- **Design**: Side-by-side layout for intuitive UX

### Screenshot 3: Voice Analysis (3.png)
**File**: `assets/pngs/3.png`

This screenshot showcases the voice analysis interface:
- **Upload Section**: WAV file upload for voice recording + optional CSV features
- **Processing**: Loading spinner during multi-model analysis
- **Ensemble Results**: 
  - CSV Model: 89% confidence
  - Audio CNN: 71% confidence
- **Final Output**: Weighted fusion (0.7×CSV + 0.3×Audio) = 85% Parkinson probability
- **Feature**: Real-time confidence visualization and model transparency

### Screenshot 4: Dashboard & History (4.png)
**File**: `assets/pngs/4.png`

This screenshot shows the user dashboard with:
- **Usage Statistics**:
  - Drawing Analyses: 7/10 used
  - Voice Analyses: 5/10 used
  - Usage Reset Timer: 1h 32m remaining
  - Last Analysis: Today 2:45 PM
- **Prediction History**: Last 3 analyses with:
  - Type (Voice/Drawing)
  - Prediction result (Parkinson/Healthy)
  - Timestamp and confidence score
  - Status indicator (✓ or ⚠)
- **Design**: Clean grid layout with color-coded status indicators

## Technical Specifications

**Image Dimensions**: 1200 × 600 pixels (16:9 aspect ratio)
**Format**: SVG (vector), PNG (raster)
**Resolution**: 96 DPI (web-standard)
**Color Scheme**: 
- Primary: Blue (#3b82f6 → #1e40af)
- Success: Green (#10b981 → #059669)
- Warning: Orange (#f59e0b → #d97706)
- Background: White with light blue gradient

## Integration with README

The README.md file references these screenshots in the "Screenshots" section:

```markdown
### Authentication Screen
![Login/Signup Interface](assets/pngs/1.png)

### Drawing Analysis
![Drawing Upload & Analysis](assets/pngs/2.png)

### Voice Analysis
![Voice Upload & Results](assets/pngs/3.png)

### Dashboard & History
![User Dashboard](assets/pngs/4.png)
```

## Notes

- SVG files are lightweight and scalable to any size
- PNG files are optimized for web display and GitHub rendering
- Screenshots use placeholder/conceptual layouts representing the actual application
- Color scheme matches the application's branding and design system
- All UI elements are simplified for clarity while maintaining design fidelity

## Future Enhancements

Consider adding:
- Actual application screenshots from the running deployment
- High-resolution versions (2K/4K)
- Dark mode theme screenshots
- Mobile responsive screenshots
- Accessibility screenshots (WCAG 2.1 compliance)

---

For more information about the application architecture and features, see the main [README.md](../README.md) file.
