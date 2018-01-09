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
      (NPresent[i, j] - NWorking[i, j]) + (NPresent[i][j+1] - NWorking[i][j+1]) <= 1;

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

  /* The following 3 constraints are needed to set NStart[n,h] to one if
  NPresent[n,h-1] and NPresent[n,h] are 0 and 1, respectively. So we
  detect the nurse traveling to the hospital at the hour h. The operation
  is just the AND of NOT NPresent[n,h-1] and NPresent[n,h]. With (NOT a)
  as (1 - a). Only the hours in [2, nHours] are used here */

  forall(n in N, h in 2..nHours) {
    NStart[n, h] >= 1 - NPresent[n, h-1] + NPresent[n, h] - 1;
    NStart[n, h] <= 1 - NPresent[n, h-1];
    NStart[n, h] <= NPresent[n, h];
  }

  /* We need to add the cases where the nurses start at hour 1. If they
  work at h, then must travel at h, otherwise not*/

  forall(n in N) {
    NStart[n, 1] == NPresent[n, 1];
  }

}
