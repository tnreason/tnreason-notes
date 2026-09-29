rm main-compilation.aux
rm main-compilation.out
pdflatex main-compilation
bibtex main-compilation
pdflatex main-compilation
open main-compilation.pdf