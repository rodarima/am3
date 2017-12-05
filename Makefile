all: report.pdf

report.pdf: report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

run: proj.mod proj.dat
	oplrun -v proj.mod proj.dat

clean:
	rm -f report.{pdf,aux,log}

.PHONY: run clean all
