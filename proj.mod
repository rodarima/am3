/*********************************************
 * OPL 12.6.0.0 Model
 *********************************************/
int nNurses=...;
int nHours=...;
range N=1..nNurses;
range H=1..nHours;
int D[d in H]=...;
int maxConsec=...;
int maxPresent=...;
int maxHours=...;
int minHours=...;

dvar boolean NPresent[i in N, s in H];
dvar boolean NWorking[i in N, w in H];
dvar boolean NStart[i in N, w in H];

minimize sum(n in N, h in H) NStart[n, h];

subject to {

	//demand per hr met
	forall(j in H)
		sum(i in N) NWorking[i, j] >= D[j];

	//Working nurses should be present
	forall(i in N, h in H)
	  	NWorking[i, h] - NPresent[i, h] <= 0;

	//nurse cannot take 2 consec rests
	forall(i in N, j in 1..nHours-1)
	  	NWorking[i, j] + NWorking[i, j+1] >= NPresent[i, j];
	
	//nurse cannot take 2 consec rests
	forall(i in N)
	  	NWorking[i, nHours] + NWorking[i, 1] >= NPresent[i, 1];

	//nurses cannot work for more than maxHours
	forall(i in N)
	  	sum(h in H)NWorking[i, h] <= maxHours;

	//Working nurses cannot work for less than minHours
	forall(i in N)
	  	sum(h in H)NWorking[i, h] >= minHours * sum(h in H)NStart[i, h];

	//nurse cannot work for more than maxConsec consecutive hrs
	forall(i in N, j in 1..nHours-maxConsec)
	  	sum(k in 0..maxConsec) NWorking[i, j+k] <= maxConsec;

	//nurses cannot be present for more than maxPresent hrs
	forall(i in N)
	  	sum(h in H)NPresent[i, h] <= maxPresent;

	//nurses can start at most once per day
	forall (n in N)
  		sum(h in H) NStart[n,h] <= 1;

	forall(n in N, h in 2..nHours)
		NStart[n, h] >= 1 - NPresent[n, h-1] + NPresent[n, h] - 1;

	forall(n in N, h in 2..nHours)
		NStart[n, h] <= 1 - NPresent[n, h-1];

	forall(n in N, h in H)
		NStart[n, h] <= NPresent[n, h];

	forall(n in N)
		NStart[n, 1] >= 1 - NPresent[n, nHours] + NPresent[n, 1] - 1;
}
