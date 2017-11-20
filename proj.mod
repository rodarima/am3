/*********************************************
 * OPL 12.6.0.0 Model
 *********************************************/
int nNurses=5;
int nHours=10;
range N=1..nNurses;
range H=1..nHours;
int D[d in H]=...;
int maxConsec=5;
int maxPresent=10;
int maxHours=8;
int minHours=2;

dvar boolean NPresent[i in N, s in H];
dvar boolean NWorking[i in N, w in H];

minimize sum(n in N, h in H) NWorking[n, h];


subject to {
	//Demand for nurses per hr
	forall(j in H)
		sum(i in N) NWorking[i, j] >= D[j];
		
	forall(i in N, h in H)
	  	NWorking[i, h] - NPresent[i, h] <= 0;
		
	//nurse cannot take 2 consec rests
	forall(i in N, j in 1..nHours-1)
	  	NWorking[i, j] + NWorking[i, j+1] >= 1*NPresent[i, j+1];
	  	
	forall(i in N)
	  	sum(h in H)NWorking[i, h] <= maxHours;
	 
	forall(i in N)
	  	sum(h in H)NWorking[i, h] >= minHours;
	
	//nurse cannot work for more than maxConsec consec hrs
	forall(i in N, j in 1..nHours-maxConsec)
	  	sum(k in 0..maxConsec) NWorking[i, j+k] <= maxConsec;
	 
	forall(i in N)
	  	sum(h in H)NPresent[i, h] <= maxPresent; 
	
}
