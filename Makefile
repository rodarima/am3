all: sol report.pdf

report.pdf: report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

proj.lp: proj.mod proj.dat
	oplrun -e proj.lp proj.mod proj.dat

sol.xml: proj.lp
	rm -f sol.xml
	cplex -f cplex-save.txt

sol: sol.xml
	python read-sol.py

clean:
	rm -f report.{pdf,aux,log} proj.lp sol.xml

.PHONY: sol clean all
