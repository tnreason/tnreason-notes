rm sub-compilation.aux
rm sub-compilation.out
pdflatex sub-compilation
bibtex sub-compilation
pdflatex sub-compilation
open sub-compilation.pdf