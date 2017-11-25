all: report.pdf

report.pdf: report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

clean:
	rm -f report.{pdf,aux,log}
