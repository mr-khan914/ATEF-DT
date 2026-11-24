#!/bin/sh
# Build the manuscript.  Usage: sh build.sh [jobname]   (default: main)
# elsarticle.cls and elsarticle-num.bst are looked up in ./texmf if they are not installed.
set -e
J=${1:-main}
export TEXINPUTS="${ATEF_TEXMF:-./texmf}//:$TEXINPUTS"
export BSTINPUTS="${ATEF_TEXMF:-./texmf}//:$BSTINPUTS"
pdflatex -interaction=nonstopmode -halt-on-error $J.tex > /dev/null
bibtex $J > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error $J.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error $J.tex > /dev/null
echo "built $J.pdf"
