cd arXiv-preprint
rm arXiv-preprint.out
rm arXiv-preprint.aux
pdflatex arXiv-qi
bibtex arXiv-qi
pdflatex arXiv-qi
open arXiv-qi.pdf