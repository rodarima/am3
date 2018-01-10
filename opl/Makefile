all: sol

proj.lp: proj.mod proj.dat
	oplrun -e proj.lp proj.mod proj.dat

sol.xml: proj.lp
	rm -f sol.xml
	cplex -f cplex-save.txt

sol: sol.xml
	python read-sol.py

clean:
	rm -f proj.lp sol.xml

.PHONY: sol clean all
