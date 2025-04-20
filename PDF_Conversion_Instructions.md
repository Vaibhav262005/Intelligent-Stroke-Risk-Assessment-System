# Converting the SRS Document to PDF

The SRS document has been created as a Markdown file (`Brain_Stroke_Prediction_System_SRS.md`). Here are several ways to convert it to a PDF document:

## Option 1: Using Pandoc (recommended for best formatting)

1. Install Pandoc:
   ```bash
   # macOS (using Homebrew)
   brew install pandoc
   brew install basictex  # or mactex for full LaTeX distribution

   # Ubuntu/Debian
   sudo apt-get install pandoc texlive-xetex
   
   # Windows
   # Download from https://pandoc.org/installing.html
   ```

2. Convert to PDF:
   ```bash
   pandoc Brain_Stroke_Prediction_System_SRS.md -o Brain_Stroke_Prediction_System_SRS.pdf --toc --toc-depth=3 --pdf-engine=xelatex -V geometry:"margin=1in"
   ```

## Option 2: Using VSCode Extensions

1. Install the "Markdown PDF" extension in VSCode
2. Open the SRS markdown file
3. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
4. Type "Markdown PDF: Export (pdf)" and press Enter

## Option 3: Using Online Markdown to PDF Converters

Several online services can convert your Markdown file to PDF:

1. [Markdown to PDF](https://www.markdowntopdf.com/)
2. [CloudConvert](https://cloudconvert.com/md-to-pdf)
3. [Dillinger](https://dillinger.io/) (Export as PDF)

## Option 4: Print to PDF from Browser

1. Open the Markdown file in a Markdown viewer (such as GitHub or any Markdown editor)
2. Use the browser's "Print" function (Cmd+P or Ctrl+P)
3. Select "Save as PDF" as the destination

## Notes for Best Results

- The document includes a detailed table of contents which works best with the Pandoc method
- Ensure your system has a PDF viewer installed to open the resulting file
- If using Pandoc, make sure you have a LaTeX distribution installed for the best formatting
- The resulting PDF should maintain all the hierarchical structure, lists, and formatting from the original document 