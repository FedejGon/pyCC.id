#!bin/bash

#pandoc paper.md -o paper.pdf
#pandoc paper.md --biblatex --bibliography=paper.bib -o paper.pdf
#pandoc paper.md --biblatex --pdf-engine=latexmk -o paper.pdf

pandoc paper.md --biblatex --pdf-engine=latexmk  --include-in-header=biblatex-config.tex -o paper.pdf
