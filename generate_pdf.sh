#!/bin/bash

# Script to generate PDF from LaTeX file
# Usage: ./generate_pdf.sh [directory_path] [latex_filename]
# Example: ./generate_pdf.sh ./3.2_SpatialDomainImageEnhancementAlgorithms/report 3.2_SpatialDomainImageEnhancementAlgorithms

# Set the script to exit on any error
set -e

# Check if directory path is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide a directory path"
    echo "Usage: $0 <directory_path> [latex_filename]"
    echo "Example: $0 ./3.2_SpatialDomainImageEnhancementAlgorithms/report 3.2_SpatialDomainImageEnhancementAlgorithms"
    exit 1
fi

# Get the target directory
TARGET_DIR="$1"
LATEX_FILE="${2:-5_frequency_domain}"

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Error: Directory '$TARGET_DIR' does not exist!"
    exit 1
fi

# Change to target directory
cd "$TARGET_DIR"

echo "🔧 Generating PDF from LaTeX file..."
echo "📁 Working directory: $(pwd)"
echo "📄 LaTeX file: ${LATEX_FILE}.tex"

# Check if LaTeX file exists
if [ ! -f "${LATEX_FILE}.tex" ]; then
    echo "❌ Error: ${LATEX_FILE}.tex not found!"
    exit 1
fi

# Update PATH to include LaTeX tools
eval "$(/usr/libexec/path_helper)"

# Determine which LaTeX compiler to use
LATEX_COMPILER=""
if command -v xelatex &> /dev/null; then
    LATEX_COMPILER="xelatex"
    echo "🚀 Compiling with XeLaTeX..."
elif command -v pdflatex &> /dev/null; then
    LATEX_COMPILER="pdflatex"
    echo "🚀 Compiling with PDFLaTeX..."
else
    echo "❌ Error: No LaTeX compiler found! Please ensure MacTeX or TeXLive is installed."
    exit 1
fi

# Run LaTeX compilation
$LATEX_COMPILER "${LATEX_FILE}.tex"

# Check if PDF was created successfully
if [ -f "${LATEX_FILE}.pdf" ]; then
    # Get file size in a human-readable format
    FILE_SIZE=$(ls -lh "${LATEX_FILE}.pdf" | awk '{print $5}')
    echo "✅ PDF generated successfully!"
    echo "📊 File: ${LATEX_FILE}.pdf"
    echo "📏 Size: $FILE_SIZE"
    
    # Count pages if pdfinfo is available
    if command -v pdfinfo &> /dev/null; then
        PAGE_COUNT=$(pdfinfo "${LATEX_FILE}.pdf" | grep "Pages:" | awk '{print $2}')
        echo "📄 Pages: $PAGE_COUNT"
    fi
    
    echo "🎉 Done! You can now open the PDF file."
else
    echo "❌ Error: PDF generation failed!"
    exit 1
fi
