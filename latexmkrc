## latexmkrc
## Build configuration for "A Study of Random Matrices".
## Run with: latexmk -pdf main

$pdf_mode = 1;                # produce PDF
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;              # run bibtex on missing references

# Clean these on `latexmk -c`. PDF is kept.
$clean_ext = "aux bbl blg fdb_latexmk fls log out synctex.gz toc";
